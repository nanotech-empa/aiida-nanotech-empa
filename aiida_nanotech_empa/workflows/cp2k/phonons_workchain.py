from copy import deepcopy

from aiida.engine import WorkChain, ExitCode
from aiida.orm import Code, Dict, Str
from aiida.orm import StructureData
from aiida.plugins import CalculationFactory, WorkflowFactory
from aiida_nanotech_empa.workflows.cp2k.cp2k_utils import (
    get_colvars_section,
    get_constraints_section,
)
from aiida_nanotech_empa.workflows.cp2k.cp2k_utils import (
    get_dft_inputs,
)

from aiida_nanotech_empa.utils import common_utils

Cp2kGeoOptWorkChain = WorkflowFactory("nanotech_empa.cp2k.geo_opt")
Cp2kCalculation = CalculationFactory("cp2k")


class Cp2kPhononsWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("code", valid_type=Code)
        spec.input("structure", valid_type=StructureData)
        spec.input("wfn_cp_commands", valid_type=Str, required=False)
        spec.input("dft_params", valid_type=Dict)
        spec.input("sys_params", valid_type=Dict)
        spec.input("phonons_params", valid_type=Dict)
        spec.input(
            "options",
            valid_type=dict,
            non_db=True,
            help="Define options for the cacluations: walltime, memory, CPUs, etc.",
        )

        # workchain outline
        spec.outline(
            cls.setup,
            cls.submit_geo_opt,
            cls.submit_phonons,
            cls.finalize,
        )
        spec.outputs.dynamic = True

        spec.exit_code(
            390,
            "ERROR_TERMINATION",
            message="One or more steps of the work chain failed.",
        )

    def setup(self):
        self.report("Inspecting input and setting up things")

        dft_params = self.inputs.dft_params.get_dict()

        (
            self.ctx.files,
            self.ctx.input_dict,
            self.ctx.structure_with_tags,
        ) = get_dft_inputs(dft_params, self.inputs.structure, "phonons_protocol.yml")
        self.ctx.sys_params = self.inputs.sys_params.get_dict()
        self.ctx.phonons_params = self.inputs.phonons_params.get_dict()
        self.ctx.input_dict["VIBRATIONAL_ANALYSIS"][
            "NPROC_REP"
        ] = self.ctx.phonons_params["nproc_rep"]

        # removal of rotations
        if "periodic" in dft_params and dft_params["periodic"] == "NONE":
            self.ctx.input_dict["VIBRATIONAL_ANALYSIS"]["FULLY_PERIODIC"] = ".FALSE."
            self.ctx.input_dict["FORCE_EVAL"]["DFT"]["PRINT"]["MOMENTS"][
                "PERIODIC"
            ] = ".FALSE."
        # constraints.
        if "constraints" in self.ctx.sys_params:
            self.ctx.input_dict["MOTION"]["CONSTRAINT"] = get_constraints_section(
                self.ctx.sys_params["constraints"]
            )
        # colvars.
        if "colvars" in self.ctx.sys_params:
            self.ctx.input_dict["FORCE_EVAL"]["SUBSYS"].update(
                get_colvars_section(self.ctx.sys_params["colvars"])
            )

        # resources
        self.ctx.options = self.inputs.options
        if self.inputs.dft_params["protocol"] == "debug":
            self.ctx.options = {
                "max_wallclock_seconds": 600,
                "resources": {
                    "num_machines": 3,
                    "num_mpiprocs_per_machine": 1,
                    "num_cores_per_mpiproc": 1,
                },
            }
        self.ctx.geo_options = deepcopy(self.ctx.options)
        self.ctx.geo_options["resources"]["num_machines"] = int(
            self.ctx.phonons_params["nproc_rep"]
            / self.ctx.options["resources"]["num_mpiprocs_per_machine"]
        )
        self.ctx.input_dict["GLOBAL"]["WALLTIME"] = max(
            600, self.ctx.options["max_wallclock_seconds"] - 600
        )
        # --------------------------------------------------

    def submit_geo_opt(self):
        """Run geo opt on the initial geometry."""

        builder = Cp2kGeoOptWorkChain.get_builder()

        builder.code = self.inputs.code
        builder.structure = self.inputs.structure
        # builder.wfn_file_path = self.inputs.wfn_file_path
        builder.dft_params = self.inputs.dft_params
        builder.sys_params = self.inputs.sys_params
        builder.options = self.ctx.geo_options
        builder.structure = StructureData(ase=self.ctx.structure_with_tags)
        builder.code = self.inputs.code

        future = self.submit(builder)
        self.report(f"Submitted geo opt of the initial geometry: {future.pk}")
        self.to_context(geo_opt=future)

    def submit_phonons(self):
        self.report("Submitting Phonons calculation")
        if not self.ctx.geo_opt.is_finished_ok:
            return self.exit_codes.ERROR_TERMINATION
        builder = Cp2kCalculation.get_builder()
        # code
        builder.code = self.inputs.code
        # structure
        builder.structure = self.ctx.geo_opt.outputs.output_structure
        # files
        # parent_folder
        builder.parent_calc_folder = self.ctx.geo_opt.outputs.remote_folder
        builder.file = self.ctx.files
        # resources
        builder.metadata.options = self.ctx.options
        # parser
        builder.metadata.options.parser_name = "cp2k_advanced_parser"
        # additional retrieved files
        builder.settings = Dict(dict={"additional_retrieve_list": ["*.eig", "*.mol"]})

        # cp2k input dictionary
        builder.parameters = Dict(self.ctx.input_dict)

        future = self.submit(builder)
        self.to_context(phonons=future)

    def finalize(self):
        self.report("Finalizing.")

        if not self.ctx.phonons.is_finished_ok:
            return self.exit_codes.ERROR_TERMINATION

        self.out("output_parameters", self.ctx.geo_opt.outputs.output_parameters)
        self.out("retrieved", self.ctx.phonons.outputs.retrieved)

        # Add the workchain pk to the input structure extras
        common_utils.add_extras(self.inputs.structure, "surfaces", self.node.uuid)

        return ExitCode(0)
