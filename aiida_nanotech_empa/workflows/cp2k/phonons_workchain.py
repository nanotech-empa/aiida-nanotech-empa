import copy

from aiida import engine, orm, plugins

from ...utils import common_utils
from . import cp2k_utils

Cp2kGeoOptWorkChain = plugins.WorkflowFactory("nanotech_empa.cp2k.geo_opt")
Cp2kCalculation = plugins.CalculationFactory("cp2k")


class Cp2kPhononsWorkChain(engine.WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("code", valid_type=orm.Code)
        spec.input("structure", valid_type=orm.StructureData)
        spec.input("parent_calc_folder", valid_type=orm.RemoteData, required=False)
        spec.input(
            "protocol",
            valid_type=orm.Str,
            default=lambda: orm.Str("standard"),
            required=False,
            help="Protocol supported by the Cp2kGeoOptWorkChain.",
        )
        spec.input("dft_params", valid_type=orm.Dict)
        spec.input("sys_params", valid_type=orm.Dict)
        spec.input("phonons_params", valid_type=orm.Dict)
        spec.input(
            "options",
            valid_type=dict,
            non_db=True,
            help="Define options for the cacluations: walltime, memory, CPUs, etc.",
        )

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
        ) = cp2k_utils.get_dft_inputs(
            dft_params,
            self.inputs.structure,
            "phonons_protocol.yml",
            protocol=self.inputs.protocol.value,
        )
        self.ctx.sys_params = self.inputs.sys_params.get_dict()
        self.ctx.phonons_params = self.inputs.phonons_params.get_dict()
        self.ctx.input_dict["VIBRATIONAL_ANALYSIS"][
            "NPROC_REP"
        ] = self.ctx.phonons_params["nproc_rep"]

        # Removal of rotations.
        if "periodic" in dft_params and dft_params["periodic"] == "NONE":
            self.ctx.input_dict["VIBRATIONAL_ANALYSIS"]["FULLY_PERIODIC"] = ".FALSE."
            self.ctx.input_dict["FORCE_EVAL"]["DFT"]["PRINT"]["MOMENTS"][
                "PERIODIC"
            ] = ".FALSE."

        # Constraints.
        if "constraints" in self.ctx.sys_params:
            self.ctx.input_dict["MOTION"][
                "CONSTRAINT"
            ] = cp2k_utils.get_constraints_section(self.ctx.sys_params["constraints"])

        # Colvars.
        if "colvars" in self.ctx.sys_params:
            self.ctx.input_dict["FORCE_EVAL"]["SUBSYS"].update(
                cp2k_utils.get_colvars_section(self.ctx.sys_params["colvars"])
            )

        # Resources.
        self.ctx.options = self.inputs.options
        self.ctx.geo_options = copy.deepcopy(self.ctx.options)
        self.ctx.geo_options["resources"]["num_machines"] = int(
            max(
                1,
                self.ctx.phonons_params["nproc_rep"]
                / self.ctx.options["resources"]["num_mpiprocs_per_machine"],
            )
        )
        self.ctx.input_dict["GLOBAL"]["WALLTIME"] = max(
            600, self.ctx.options["max_wallclock_seconds"] - 600
        )

    def submit_geo_opt(self):
        """Run geo opt on the initial geometry."""

        builder = Cp2kGeoOptWorkChain.get_builder()

        builder.code = self.inputs.code
        builder.structure = self.inputs.structure

        # Restart WFN.
        if "parent_calc_folder" in self.inputs:
            builder.parent_calc_folder = self.inputs.parent_calc_folder
        builder.protocol = self.inputs.protocol
        builder.dft_params = self.inputs.dft_params
        builder.sys_params = self.inputs.sys_params
        builder.options = self.ctx.geo_options
        builder.structure = orm.StructureData(ase=self.ctx.structure_with_tags)
        builder.code = self.inputs.code

        future = self.submit(builder)
        self.report(f"Submitted geo opt of the initial geometry: {future.pk}")
        self.to_context(geo_opt=future)

    def submit_phonons(self):
        self.report("Submitting Phonons calculation")
        if not self.ctx.geo_opt.is_finished_ok:
            return self.exit_codes.ERROR_TERMINATION
        builder = Cp2kCalculation.get_builder()
        builder.code = self.inputs.code
        builder.structure = self.ctx.geo_opt.outputs.output_structure
        builder.parent_calc_folder = self.ctx.geo_opt.outputs.remote_folder
        builder.file = self.ctx.files
        builder.metadata.options = self.ctx.options
        builder.metadata.options.parser_name = "cp2k_advanced_parser"
        builder.settings = orm.Dict(
            dict={"additional_retrieve_list": ["*.eig", "*.mol"]}
        )
        builder.parameters = orm.Dict(self.ctx.input_dict)

        future = self.submit(builder)
        self.to_context(phonons=future)

    def finalize(self):
        self.report("Finalizing.")

        if not self.ctx.phonons.is_finished_ok:
            return self.exit_codes.ERROR_TERMINATION

        self.out("output_parameters", self.ctx.geo_opt.outputs.output_parameters)
        self.out("retrieved", self.ctx.phonons.outputs.retrieved)

        # Add the workchain pk to the input structure extras.
        common_utils.add_extras(self.inputs.structure, "surfaces", self.node.uuid)

        return engine.ExitCode(0)
