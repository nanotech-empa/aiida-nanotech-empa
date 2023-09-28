import os

import numpy as np
from aiida import engine, orm, plugins

from ...utils import common_utils
from . import cp2k_utils
from .diag_workchain import Cp2kDiagWorkChain

AfmCalculation = plugins.CalculationFactory("nanotech_empa.afm")


class Cp2kAfmWorkChain(engine.WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("cp2k_code", valid_type=orm.Code)
        spec.input("structure", valid_type=orm.StructureData)
        spec.input("parent_calc_folder", valid_type=orm.RemoteData, required=False)
        spec.input(
            "protocol",
            valid_type=orm.Str,
            default=lambda: orm.Str("standard"),
            required=False,
            help="Protocol supported by the Cp2kDiagWorkChain.",
        )
        spec.input("dft_params", valid_type=orm.Dict)
        spec.input(
            "options",
            valid_type=dict,
            non_db=True,
            help="Define options for the cacluations: walltime, memory, CPUs, etc.",
        )

        spec.input("afm_pp_code", valid_type=orm.Code)
        spec.input("afm_pp_params", valid_type=orm.Dict)
        spec.input("afm_2pp_code", valid_type=orm.Code)
        spec.input("afm_2pp_params", valid_type=orm.Dict)

        spec.outline(
            cls.setup,
            cls.run_diag_scf,
            cls.run_afms,
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
        ase_geom.set_tags(np.zeros(len(ase_geom)))
        n_atoms = len(structure.sites)
        self.ctx.dft_params = self.inputs.dft_params.get_dict()
        if "smear_t" in self.ctx.dft_params:
            added_mos = np.max([100, int(1.2 * n_atoms * 2 / 5.0)])
            self.ctx.dft_params["added_mos"] = added_mos
        self.ctx.files = {
            "geo_no_labels": cp2k_utils.make_geom_file(ase_geom, "geom.xyz"),
            "pp": orm.SinglefileData(
                file=os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    ".",
                    "data",
                    "atomtypes_pp.ini",
                )
            ),
            "2pp": orm.SinglefileData(
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
        builder.protocol = self.inputs.protocol
        builder.dft_params = orm.Dict(self.ctx.dft_params)
        builder.options = orm.Dict(self.inputs.options)

        # Restart wfn.
        if "parent_calc_folder" in self.inputs:
            builder.parent_calc_folder = self.inputs.parent_calc_folder

        future = self.submit(builder)
        self.to_context(diag_scf=future)

    def run_afms(self):
        self.report("Running PP")
        if not common_utils.check_if_calc_ok(self, self.ctx.diag_scf):
            return self.exit_codes.ERROR_TERMINATION
        afm_pp_inputs = {}

        afm_pp_inputs["geo_no_labels"] = self.ctx.files["geo_no_labels"]
        afm_pp_inputs["metadata"] = {}
        afm_pp_inputs["metadata"]["label"] = "afm_pp"
        afm_pp_inputs["code"] = self.inputs.afm_pp_code
        afm_pp_inputs["parameters"] = self.inputs.afm_pp_params
        afm_pp_inputs["parent_calc_folder"] = self.ctx.diag_scf.outputs.remote_folder
        afm_pp_inputs["atomtypes"] = self.ctx.files["pp"]
        afm_pp_inputs["metadata"]["options"] = {
            "max_wallclock_seconds": 86000,
            "resources": {
                "num_machines": 1,
                "num_mpiprocs_per_machine": 1,
                "num_cores_per_mpiproc": 1,
            },
        }
        self.report("Afm pp inputs: " + str(afm_pp_inputs))
        afm_pp_future = self.submit(AfmCalculation, **afm_pp_inputs)
        self.to_context(afm_pp=afm_pp_future)

        self.report("Running 2PP")

        afm_2pp_inputs = {}
        afm_2pp_inputs["geo_no_labels"] = self.ctx.files["geo_no_labels"]
        afm_2pp_inputs["metadata"] = {}
        afm_2pp_inputs["metadata"]["label"] = "afm_2pp"
        afm_2pp_inputs["code"] = self.inputs.afm_2pp_code
        afm_2pp_inputs["parameters"] = self.inputs.afm_2pp_params
        afm_2pp_inputs["parent_calc_folder"] = self.ctx.diag_scf.outputs.remote_folder
        afm_2pp_inputs["atomtypes"] = self.ctx.files["2pp"]
        afm_2pp_inputs["metadata"]["options"] = {
            "max_wallclock_seconds": 86000,
            "resources": {
                "num_machines": 1,
                "num_mpiprocs_per_machine": 1,
                "num_cores_per_mpiproc": 1,
            },
        }
        self.report("Afm 2pp inputs: " + str(afm_2pp_inputs))
        afm_2pp_future = self.submit(AfmCalculation, **afm_2pp_inputs)
        self.to_context(afm_2pp=afm_2pp_future)

    def finalize(self):
        retrieved_list = [
            obj.name for obj in self.ctx.afm_pp.outputs.retrieved.list_objects()
        ]
        pp_worked = "df.npy" in retrieved_list and "df_vec.npy" in retrieved_list
        retrieved_list = [
            obj.name for obj in self.ctx.afm_2pp.outputs.retrieved.list_objects()
        ]
        pp2_worked = "df.npy" in retrieved_list and "df_vec.npy" in retrieved_list
        if not pp_worked or not pp2_worked:
            self.report("AFM calculation did not finish correctly")
            return self.exit_codes.ERROR_TERMINATION

        # Add the workchain pk to the input structure extras.
        common_utils.add_extras(self.inputs.structure, "surfaces", self.node.uuid)
        self.report("Work chain is finished")
