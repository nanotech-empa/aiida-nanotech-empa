import numpy as np
from aiida import engine, orm, plugins
from aiida_shell import launch_shell_job

from ...utils import common_utils

Cp2kDiagWorkChain = plugins.WorkflowFactory("nanotech_empa.cp2k.diag")
StmCalculation = plugins.CalculationFactory("nanotech_empa.stm")


class Cp2kOrbitalsWorkChain(engine.WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)

        # Codes.
        spec.input("cp2k_code", valid_type=orm.Code)
        spec.input("spm_code", valid_type=orm.Code)
        spec.input("cubehandler_code", valid_type=orm.Code, required=False)

        # Inputs.
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
        spec.input("spm_params", valid_type=orm.Dict)
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
            engine.if_(cls.should_run_cubehandler)(
                cls.run_cubehandler,
            ),
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
        n_lumo = int(self.inputs.spm_params.get_dict()["--n_lumo"])
        added_mos = np.max([n_lumo, 20])
        self.ctx.dft_params = self.inputs.dft_params.get_dict()
        self.ctx.dft_params["added_mos"] = added_mos

    def run_diag_scf(self):
        self.report("Running CP2K diagonalization SCF")
        builder = Cp2kDiagWorkChain.get_builder()
        builder.cp2k_code = self.inputs.cp2k_code
        builder.structure = self.inputs.structure
        builder.dft_params = orm.Dict(self.ctx.dft_params)
        builder.protocol = self.inputs.protocol

        # Restart wfn.
        if "parent_calc_folder" in self.inputs:
            builder.parent_calc_folder = self.inputs.parent_calc_folder
        builder.settings = orm.Dict(
            {
                "additional_retrieve_list": [
                    "aiida.inp",
                    "BASIS_MOLOPT",
                    "aiida.coords.xyz",
                    "aiida-RESTART.wfn",
                ]
            }
        )
        builder.options = orm.Dict(self.inputs.options)

        future = self.submit(builder)
        self.to_context(diag_scf=future)

    def run_stm(self):
        self.report("STM calculation")
        if not common_utils.check_if_calc_ok(self, self.ctx.diag_scf):
            return self.exit_codes.ERROR_TERMINATION

        inputs = {}
        inputs["metadata"] = {}
        inputs["metadata"]["label"] = "orb"
        inputs["code"] = self.inputs.spm_code
        inputs["parameters"] = self.inputs.spm_params
        inputs["parent_calc_folder"] = self.ctx.diag_scf.outputs.remote_folder
        inputs["metadata"]["options"] = {
            "resources": {
                "num_machines": 1,
                "num_mpiprocs_per_machine": min(
                    36,
                    self.inputs.cp2k_code.computer.get_default_mpiprocs_per_machine(),
                ),
                "num_cores_per_mpiproc": 1,
            },
            "max_wallclock_seconds": 7200,
        }

        # Need to make an explicit instance for the node to be stored to aiida.
        settings = orm.Dict({"additional_retrieve_list": ["orb.npz"]})
        inputs["settings"] = settings

        future = self.submit(StmCalculation, **inputs)
        return engine.ToContext(stm=future)

    def should_run_cubehandler(self):
        return "cubehandler_code" in self.inputs

    def run_cubehandler(self):
        self.report("Running CubeHandler")
        if not common_utils.check_if_calc_ok(self, self.ctx.diag_scf):
            return self.exit_codes.ERROR_TERMINATIONx

        _, node = launch_shell_job(
            self.inputs.cubehandler_code,
            arguments=["shrink", ".", "out_cubes"],
            metadata={
                "options": {
                    "prepend_text": "conda activate cubehandler",
                    "use_symlinks": True,
                },
                "computer": orm.load_computer("daint-gpu"),
                "label": "cube-shrink",
            },
            nodes={"remote_previous_job": self.ctx.diag_scf.outputs.remote_folder},
            outputs=["out_cubes"],
        )
        self.ctx.cubehandler_uuid = node.uuid

    def finalize(self):
        if "orb.npz" not in [
            obj.name for obj in self.ctx.stm.outputs.retrieved.list_objects()
        ]:
            self.report("Orbital calculation did not finish correctly")
            return self.exit_codes.ERROR_TERMINATION
        self.out("dft_output_parameters", self.ctx.diag_scf.outputs.output_parameters)
        self.out("retrieved", self.ctx.diag_scf.outputs.retrieved)
        common_utils.add_extras(self.inputs.structure, "surfaces", self.node.uuid)
        if "cubehandler_code" in self.inputs:
            common_utils.add_extras(
                self.inputs.structure, "surfaces", self.ctx.cubehandler_uuid
            )
        self.report("The workchain is finished")
