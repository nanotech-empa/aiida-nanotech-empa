import copy

import numpy as np
from aiida import engine, orm, plugins

from ...utils import common_utils
from . import cp2k_utils
from .geo_opt_workchain import validate_on_unhandled_failure

Cp2kBaseWorkChain = plugins.WorkflowFactory("cp2k.base")


class Cp2kDiagWorkChain(engine.WorkChain):
    @classmethod
    def define_common_inputs(cls, spec):
        spec.input("cp2k_code", valid_type=orm.Code)
        spec.input("structure", valid_type=orm.StructureData)
        spec.input("parent_calc_folder", valid_type=orm.RemoteData, required=False)
        spec.input("dft_params", valid_type=orm.Dict, default=lambda: orm.Dict(dict={}))
        spec.input(
            "protocol",
            valid_type=orm.Str,
            default=lambda: orm.Str("standard"),
            required=False,
            help="Protocol supported by the Cp2kBaseWorkChain.",
        )
        spec.input("pdos_lists", valid_type=orm.List, required=False)
        spec.input("settings", valid_type=orm.Dict, required=False)
        spec.input(
            "options",
            valid_type=orm.Dict,
            default=lambda: orm.Dict(
                dict={
                    "max_wallclock_seconds": 600,
                    "resources": {
                        "num_machines": 1,
                        "num_mpiprocs_per_machine": 1,
                        "num_cores_per_mpiproc": 1,
                    },
                }
            ),
            required=False,
            help="Define options for the cacluations: walltime, memory, CPUs, etc.",
        )
        spec.input(
            "max_iterations",
            valid_type=orm.Int,
            default=lambda: orm.Int(5),
            required=False,
            help="Maximum number of CP2K restart attempts delegated to cp2k.base.",
        )
        spec.input(
            "clean_workdir",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            required=False,
            help="Clean called CP2K calculation work directories after termination.",
        )
        spec.input(
            "on_unhandled_failure",
            valid_type=orm.Str,
            default=lambda: orm.Str("pause"),
            required=False,
            validator=validate_on_unhandled_failure,
            help="Action for unhandled cp2k.base failures: abort, pause, restart_once, or restart_and_pause.",
        )
        spec.input(
            "pause_on_max_iterations",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(True),
            required=False,
            help="Pause cp2k.base for inspection when restart max_iterations is reached.",
        )

    @classmethod
    def define_common_outputs(cls, spec):
        spec.outputs.dynamic = True

        spec.exit_code(
            390,
            "ERROR_TERMINATION",
            message="One or more steps of the work chain failed.",
        )

    @classmethod
    def define(cls, spec):
        super().define(spec)
        cls.define_common_inputs(spec)
        spec.outline(
            cls.setup,
            cls.run_ot_scf,
            cls.run_diag_scf,
            cls.finalize,
        )
        cls.define_common_outputs(spec)

    def set_restart_policy(self, builder):
        builder.max_iterations = self.inputs.max_iterations
        builder.clean_workdir = self.inputs.clean_workdir
        builder.on_unhandled_failure = self.inputs.on_unhandled_failure
        builder.pause_on_max_iterations = self.inputs.pause_on_max_iterations

    def setup(self):
        self.report("Setting up workchain")
        self.ctx.dft_params = self.inputs.dft_params.get_dict()
        self.ctx.files = cp2k_utils.get_dft_file_inputs(self.ctx.dft_params)

        structure = self.inputs.structure
        self.ctx.n_atoms = len(structure.sites)

        self.ctx.dft_params.setdefault("periodic", "XYZ")
        self.ctx.dft_params.setdefault("uks", False)
        self.ctx.dft_params.setdefault("elpa_switch", False)
        self.ctx.dft_params.setdefault("sc_diag", False)

        # Resources.
        self.ctx.options = self.inputs.options.get_dict()

        # Get cutoff.
        self.ctx.cutoff = cp2k_utils.get_cutoff(structure=structure)

        # Overwrite cutoff if given in dft_params.
        if "cutoff" in self.ctx.dft_params:
            self.ctx.cutoff = self.ctx.dft_params["cutoff"]

        # Get initial magnetization.
        magnetization_per_site = [0 for i in range(len(self.inputs.structure.sites))]
        if "uks" in self.ctx.dft_params and self.ctx.dft_params["uks"]:
            magnetization_per_site = self.ctx.dft_params["magnetization_per_site"]

        structure_with_tags, kinds_dict = cp2k_utils.determine_kinds(
            structure, magnetization_per_site
        )
        self.ctx.structure_with_tags = structure_with_tags.get_ase()
        self._handle_periodicity(self.ctx.structure_with_tags)

        self.ctx.kinds_section = cp2k_utils.get_kinds_section(
            kinds_dict, protocol="gpw", dft_params=self.ctx.dft_params
        )

    def _handle_periodicity(self, structure):
        if self.ctx.dft_params["periodic"] == "NONE":
            # Make sure cell is big enough for MT poisson solver and center positions.
            if self.inputs.protocol == "debug":
                extra_cell = 9.0  # Angstrom.
            else:
                extra_cell = 15.0  # Angstrom.
            structure.cell = 2 * (np.ptp(structure.positions, axis=0)) + extra_cell
            structure.center()

    def run_ot_scf(self):
        self.report("Running CP2K OT SCF")

        # Load input template.
        input_dict = cp2k_utils.load_protocol(
            "scf_ot_protocol.yml", self.inputs.protocol.value
        )
        cp2k_utils.apply_dft_file_names(input_dict, self.ctx.dft_params)

        # Set workflow inputs.
        builder = Cp2kBaseWorkChain.get_builder()
        builder.cp2k.code = self.inputs.cp2k_code
        self.set_restart_policy(builder)
        builder.cp2k.structure = orm.StructureData(ase=self.ctx.structure_with_tags)

        builder.cp2k.file = self.ctx.files
        # restart wfn
        if "parent_calc_folder" in self.inputs:
            builder.cp2k.parent_calc_folder = self.inputs.parent_calc_folder

        if "charge" in self.ctx.dft_params:
            input_dict["FORCE_EVAL"]["DFT"]["CHARGE"] = self.ctx.dft_params["charge"]
        if not self.ctx.dft_params.get("vdw", False):
            input_dict["FORCE_EVAL"]["DFT"]["XC"].pop("VDW_POTENTIAL", None)
        cp2k_utils.apply_xc_settings(input_dict, self.ctx.dft_params)

        # POISSON_SOLVER
        if self.ctx.dft_params["periodic"] == "NONE":
            input_dict["FORCE_EVAL"]["DFT"]["POISSON"]["PERIODIC"] = "NONE"
            input_dict["FORCE_EVAL"]["DFT"]["POISSON"]["POISSON_SOLVER"] = "MT"

        # UKS
        if self.ctx.dft_params["uks"]:
            input_dict["FORCE_EVAL"]["DFT"]["UKS"] = ".TRUE."
            input_dict["FORCE_EVAL"]["DFT"]["MULTIPLICITY"] = self.ctx.dft_params[
                "multiplicity"
            ]

        # CUTOFF.
        input_dict["FORCE_EVAL"]["DFT"]["MGRID"]["CUTOFF"] = self.ctx.cutoff

        # KINDS section
        cp2k_utils.dict_merge(input_dict, self.ctx.kinds_section)

        # Setup walltime.
        input_dict["GLOBAL"]["WALLTIME"] = max(
            600, self.ctx.options["max_wallclock_seconds"] - 600
        )

        # Switch on restart_incomplete_calculation handler disabled by default.
        builder.handler_overrides = orm.Dict(
            {"restart_incomplete_calculation": {"enabled": True}}
        )

        builder.cp2k.metadata.options = self.ctx.options

        # Parser.
        builder.cp2k.metadata.options.parser_name = "cp2k_advanced_parser"

        self.update_ot_input_dict(input_dict)

        # CP2K input dictionary.
        builder.cp2k.parameters = orm.Dict(input_dict)
        self.ctx.input_dict = copy.deepcopy(input_dict)

        future = self.submit(builder)
        self.to_context(ot_scf=future)

    def update_ot_input_dict(self, input_dict):
        pass

    def run_diag_scf(self):
        self.report("Running CP2K diagonalization SCF")
        if not common_utils.check_if_calc_ok(self, self.ctx.ot_scf):
            return self.exit_codes.ERROR_TERMINATION

        # Load input template.
        scf_dict = cp2k_utils.load_protocol(
            "scf_diag_protocol.yml", self.inputs.protocol.value
        )

        input_dict = copy.deepcopy(self.ctx.input_dict)
        if self.ctx.dft_params["elpa_switch"]:
            input_dict["GLOBAL"]["PREFERRED_DIAG_LIBRARY"] = "ELPA"
            input_dict["GLOBAL"]["ELPA_KERNEL"] = "AVX2_BLOCK2"
            input_dict["GLOBAL"]["DBCSR"] = {"USE_MPI_ALLOCATOR": ".FALSE."}
        input_dict["FORCE_EVAL"]["DFT"].pop("SCF")
        input_dict["FORCE_EVAL"]["DFT"]["SCF"] = scf_dict
        cp2k_utils.apply_xc_settings(input_dict, self.ctx.dft_params, scf_method="diag")
        if "added_mos" in self.ctx.dft_params:
            input_dict["FORCE_EVAL"]["DFT"]["SCF"]["ADDED_MOS"] = self.ctx.dft_params[
                "added_mos"
            ]

        # PDOS
        if "pdos_lists" in self.inputs:
            pdos_list_dicts = [
                {"COMPONENTS": "", "LIST": e} for e in self.inputs.pdos_lists
            ]
            input_dict["FORCE_EVAL"]["DFT"]["PRINT"]["PDOS"] = {
                "NLUMO": self.ctx.dft_params["added_mos"],
                "LDOS": pdos_list_dicts,
            }

        smearing = "smear_t" in self.ctx.dft_params
        if smearing and self.ctx.dft_params["sc_diag"]:
            input_dict["FORCE_EVAL"]["DFT"]["SCF"]["SMEAR"][
                "ELECTRONIC_TEMPERATURE"
            ] = self.ctx.dft_params["smear_t"]
        else:
            input_dict["FORCE_EVAL"]["DFT"]["SCF"].pop("SMEAR")

        # UKS
        if (
            self.ctx.dft_params["uks"]
            and smearing
            and self.ctx.dft_params["force_multiplicity"]
        ):
            input_dict["FORCE_EVAL"]["DFT"]["SCF"]["SMEAR"]["FIXED_MAGNETIC_MOMENT"] = (
                self.ctx.dft_params["multiplicity"] - 1
            )
        # no self consistent diag
        if not self.ctx.dft_params["sc_diag"]:
            if (
                "SMEAR" in input_dict["FORCE_EVAL"]["DFT"]["SCF"]
            ):  # could have been already removed if smear false and sc_diag true
                input_dict["FORCE_EVAL"]["DFT"]["SCF"].pop("SMEAR")
            input_dict["FORCE_EVAL"]["DFT"]["SCF"]["EPS_SCF"] = "1.0E-1"
            input_dict["FORCE_EVAL"]["DFT"]["SCF"]["OUTER_SCF"]["EPS_SCF"] = "1.0E-1"

        # Printing orbitals.
        if "nhomo" in self.ctx.dft_params:
            input_dict["FORCE_EVAL"]["DFT"]["PRINT"]["MO_CUBES"]["NHOMO"] = (
                self.ctx.dft_params["nhomo"]
            )
            input_dict["FORCE_EVAL"]["DFT"]["PRINT"]["MO_CUBES"]["STRIDE"] = "2 2 2"
        if "nlumo" in self.ctx.dft_params:
            input_dict["FORCE_EVAL"]["DFT"]["PRINT"]["MO_CUBES"]["NLUMO"] = (
                self.ctx.dft_params["nlumo"]
            )
            input_dict["FORCE_EVAL"]["DFT"]["PRINT"]["MO_CUBES"]["STRIDE"] = "2 2 2"

        self.update_diag_input_dict(input_dict)

        # Setup walltime.
        input_dict["GLOBAL"]["WALLTIME"] = max(
            600, self.ctx.options["max_wallclock_seconds"] - 600
        )

        builder = Cp2kBaseWorkChain.get_builder()
        builder.cp2k.code = self.inputs.cp2k_code
        self.set_restart_policy(builder)
        builder.cp2k.structure = orm.StructureData(ase=self.ctx.structure_with_tags)

        builder.cp2k.file = self.ctx.files
        if "settings" in self.inputs:
            builder.cp2k.settings = self.inputs.settings

        builder.cp2k.parent_calc_folder = self.ctx.ot_scf.outputs.remote_folder

        # Switch on restart_incomplete_calculation handler disabled by default.
        builder.handler_overrides = orm.Dict(
            {"restart_incomplete_calculation": {"enabled": True}}
        )

        builder.cp2k.metadata.options = self.ctx.options

        # Use the advanced parser.
        builder.cp2k.metadata.options.parser_name = "cp2k_advanced_parser"

        # CP2K input dictionary.
        builder.cp2k.parameters = orm.Dict(input_dict)

        self.to_context(diag_scf=self.submit(builder))

    def finalize(self):
        if not common_utils.check_if_calc_ok(self, self.ctx.diag_scf):
            self.report("diagonalization scf failed")
            return self.exit_codes.ERROR_TERMINATION

        self.out("output_parameters", self.ctx.diag_scf.outputs.output_parameters)
        self.out("remote_folder", self.ctx.diag_scf.outputs.remote_folder)
        self.out("retrieved", self.ctx.diag_scf.outputs.retrieved)
        self.report("Work chain is finished")

    def update_diag_input_dict(self, input_dict):
        """Hook for derived workchains to add diagonalization-only CP2K input."""
