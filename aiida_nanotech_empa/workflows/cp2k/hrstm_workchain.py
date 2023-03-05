import os
import numpy as np

from aiida.orm import StructureData
from aiida.orm import Dict
from aiida.orm import Str
from aiida.orm import SinglefileData
from aiida.orm import Code

from aiida.engine import WorkChain, ToContext

from aiida_nanotech_empa.utils import common_utils

from aiida.plugins import CalculationFactory, WorkflowFactory
from aiida_nanotech_empa.workflows.cp2k.cp2k_utils import make_geom_file

Cp2kDiagWorkChain = WorkflowFactory("nanotech_empa.cp2k.diag")
HrstmCalculation = CalculationFactory("nanotech_empa.hrstm")
AfmCalculation = CalculationFactory("nanotech_empa.afm")


class Cp2kHrstmWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super(Cp2kHrstmWorkChain, cls).define(spec)

        spec.input("cp2k_code", valid_type=Code)
        spec.input("structure", valid_type=StructureData)
        spec.input("wfn_file_path", valid_type=Str, default=Str(""))
        spec.input("dft_params", valid_type=Dict)
        spec.input(
            "options",
            valid_type=dict,
            non_db=True,
            help="Define options for the cacluations: walltime, memory, CPUs, etc.",
        )

        spec.input("ppm_code", valid_type=Code)
        spec.input("ppm_params", valid_type=Dict)

        spec.input("hrstm_code", valid_type=Code)
        spec.input("hrstm_params", valid_type=Dict)

        spec.outline(
            cls.setup,
            cls.run_diag_scf,
            cls.run_ppm,
            cls.run_hrstm,
            cls.finalize,
        )

        spec.outputs.dynamic = True
        spec.exit_code(
            390,
            "ERROR_TERMINATION",
            message="One or more steps of the work chain failed.",
        )

    def setup(self):
        self.report("Setting up workchain")
        structure = self.inputs.structure
        ase_geom = structure.get_ase()
        n_atoms = len(structure.sites)
        emax = 2
        self.ctx.options = self.inputs.options
        self.ctx.dft_params = self.inputs.dft_params.get_dict()
        added_mos = np.max([100, int(1.2 * n_atoms * emax / 5.0)])
        self.ctx.dft_params["added_mos"] = added_mos

        self.ctx.files = {
            "geo_no_labels": make_geom_file(ase_geom, "geom.xyz"),
            "2pp": SinglefileData(
                file=os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    ".",
                    "data",
                    "atomtypes_2pp.ini",
                )
            ),
        }

    def run_diag_scf(self):
        self.report("Running CP2K diagonalization SCF")
        builder = Cp2kDiagWorkChain.get_builder()
        builder.cp2k_code = self.inputs.cp2k_code
        builder.structure = self.inputs.structure
        builder.dft_params = Dict(self.ctx.dft_params)
        builder.options = Dict(self.ctx.options)

        future = self.submit(builder)
        self.to_context(diag_scf=future)

    def run_ppm(self):
        self.report("Running PPM")
        if not common_utils.check_if_calc_ok(self, self.ctx.diag_scf):
            return self.exit_codes.ERROR_TERMINATION  # pylint: disable=no-member
        inputs = {}
        inputs["geo_no_labels"] = self.ctx.files["geo_no_labels"]
        inputs["metadata"] = {}
        inputs["metadata"]["label"] = "hrstm_ppm"
        inputs["code"] = self.inputs.ppm_code
        inputs["parameters"] = self.inputs.ppm_params
        inputs["parent_calc_folder"] = self.ctx.diag_scf.outputs.remote_folder
        # TODO set atom types properly
        inputs["atomtypes"] = self.ctx.files["2pp"]
        inputs["metadata"]["options"] = {
            "max_wallclock_seconds": 21600,
            "resources": {
                "num_machines": 1,
                "num_mpiprocs_per_machine": 1,
                "num_cores_per_mpiproc": 1,
            },
        }
        self.report("PPM inputs: " + str(inputs))

        future = self.submit(AfmCalculation, **inputs)
        return ToContext(ppm=future)

    def run_hrstm(self):
        self.report("Running HR-STM")
        retrieved_list = [
            obj.name for obj in self.ctx.ppm.outputs.retrieved.list_objects()
        ]
        pp_worked = "df.npy" in retrieved_list and "df_vec.npy" in retrieved_list
        if not pp_worked:
            self.report("AFM calculation did not finish correctly")
            return self.exit_codes.ERROR_TERMINATION
        inputs = {}
        inputs["metadata"] = {}
        inputs["metadata"]["label"] = "hrstm"
        inputs["code"] = self.inputs.hrstm_code
        inputs["parameters"] = self.inputs.hrstm_params
        inputs["parent_calc_folder"] = self.ctx.diag_scf.outputs.remote_folder
        inputs["ppm_calc_folder"] = self.ctx.ppm.outputs.remote_folder
        inputs["metadata"]["options"] = {
            "resources": {"num_machines": 8, "num_mpiprocs_per_machine": 1},
            "max_wallclock_seconds": 72000,  # 20:00 hours
        }

        self.report("HR-STM Inputs: " + str(inputs))

        future = self.submit(HrstmCalculation, **inputs)
        return ToContext(hrstm=future)

    def finalize(self):
        self.report("Work chain is finished")
        retrieved_list = [
            obj.name for obj in self.ctx.hrstm.outputs.retrieved.list_objects()
        ]
        hrstm_worked = "hrstm_meta.npy" in retrieved_list
        if not hrstm_worked:
            self.report("HRSTM calculation did not finish correctly")
            return self.exit_codes.ERROR_TERMINATION
        # Add the workchain pk to the input structure extras
        common_utils.add_extras(self.inputs.structure, "surfaces", self.node.uuid)
        self.report("Work chain is finished")
