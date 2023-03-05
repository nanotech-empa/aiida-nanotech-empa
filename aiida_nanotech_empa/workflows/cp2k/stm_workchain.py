import numpy as np

from aiida.engine import ToContext, WorkChain
from aiida.orm import Code, Dict, Str, StructureData

from aiida.plugins import CalculationFactory, WorkflowFactory

from aiida_nanotech_empa.utils import common_utils


Cp2kDiagWorkChain = WorkflowFactory("nanotech_empa.cp2k.diag")
StmCalculation = CalculationFactory("nanotech_empa.stm")


class Cp2kStmWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super(Cp2kStmWorkChain, cls).define(spec)

        spec.input("cp2k_code", valid_type=Code)
        spec.input("structure", valid_type=StructureData)
        spec.input("wfn_file_path", valid_type=Str, required=False)
        spec.input("dft_params", valid_type=Dict)
        spec.input("spm_code", valid_type=Code)
        spec.input("spm_params", valid_type=Dict)
        spec.input(
            "options",
            valid_type=dict,
            non_db=True,
            help="Define options for the cacluations: walltime, memory, CPUs, etc.",
        )

        spec.outline(
            cls.setup,
            cls.run_diag_scf,
            cls.run_stm,
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
        self.ctx.n_atoms = len(structure.sites)
        emax = float(self.inputs.spm_params.get_dict()["--energy_range"][1])
        added_mos = np.max([100, int(1.2 * self.ctx.n_atoms * emax / 5.0)])
        self.ctx.dft_params = self.inputs.dft_params.get_dict()
        self.ctx.dft_params["added_mos"] = added_mos

    def run_diag_scf(self):
        self.report("Running CP2K diagonalization SCF")
        builder = Cp2kDiagWorkChain.get_builder()
        builder.cp2k_code = self.inputs.cp2k_code
        builder.structure = self.inputs.structure
        builder.dft_params = Dict(self.ctx.dft_params)
        builder.options = Dict(self.inputs.options)

        future = self.submit(builder)
        self.to_context(diag_scf=future)

    def run_stm(self):
        self.report("STM calculation")
        if not common_utils.check_if_calc_ok(self, self.ctx.diag_scf):
            return self.exit_codes.ERROR_TERMINATION  # pylint: disable=no-member
        inputs = {}
        inputs["metadata"] = {}
        inputs["metadata"]["label"] = "stm"
        inputs["code"] = self.inputs.spm_code
        inputs["parameters"] = self.inputs.spm_params
        inputs["parent_calc_folder"] = self.ctx.diag_scf.outputs.remote_folder

        n_machines = 6
        if self.ctx.n_atoms > 1000:
            n_machines = 12
        if self.ctx.n_atoms > 2000:
            n_machines = 18
        if self.ctx.n_atoms > 3000:
            n_machines = 24
        if self.ctx.n_atoms > 4000:
            n_machines = 30

        inputs["metadata"]["options"] = {
            "resources": {"num_machines": n_machines},
            "max_wallclock_seconds": 36000,
        }
        if self.inputs.dft_params["protocol"] == "debug":
            inputs["metadata"]["options"]["max_wallclock_seconds"] = 600
        # Need to make an explicit instance for the node to be stored to aiida
        settings = Dict({"additional_retrieve_list": ["stm.npz"]})
        inputs["settings"] = settings

        future = self.submit(StmCalculation, **inputs)
        return ToContext(stm=future)

    def finalize(self):
        if "stm.npz" not in [
            obj.name for obj in self.ctx.stm.outputs.retrieved.list_objects()
        ]:
            self.report("STM calculation did not finish correctly")
            return self.exit_codes.ERROR_TERMINATION

        self.out("dft_output_parameters", self.ctx.diag_scf.outputs.output_parameters)

        # Add the workchain pk to the input structure extras
        common_utils.add_extras(self.inputs.structure, "surfaces", self.node.uuid)
        self.report("Work chain is finished")
