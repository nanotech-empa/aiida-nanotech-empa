import copy
import pathlib

import numpy as np
from aiida import engine, orm
from aiida_cp2k.calculations import Cp2kCalculation

from . import cp2k_utils

ALLOWED_PROTOCOLS = ["gapw_std", "gapw_hq", "gpw_std"]


class Cp2kMoleculeGwWorkChain(engine.WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("code", valid_type=orm.Code)
        spec.input("structure", valid_type=orm.StructureData)

        spec.input(
            "protocol",
            valid_type=orm.Str,
            default=lambda: orm.Str("gapw_std"),
            required=False,
            help="Either 'gapw_std', 'gapw_hq', 'gpw_std'",
        )

        spec.input(
            "run_image_charge",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            required=False,
            help="Run the image charge correction calculation.",
        )
        spec.input(
            "z_ic_plane",
            valid_type=orm.Float,
            default=lambda: orm.Float(8.22),
            required=False,
        )

        spec.input(
            "multiplicity",
            valid_type=orm.Int,
            default=lambda: orm.Int(0),
            required=False,
        )
        spec.input(
            "magnetization_per_site",
            valid_type=orm.List,
            default=lambda: orm.List(list=[]),
            required=False,
        )
        spec.input_namespace(
            "options",
            valid_type=dict,
            non_db=True,
            required=False,
            help="Define options for the cacluations: walltime, memory, CPUs, etc.",
        )

        spec.input(
            "options.scf",
            valid_type=dict,
            non_db=True,
            required=False,
            help="Define options for the SCF cacluation: walltime, memory, CPUs, etc.",
        )

        spec.input(
            "options.gw",
            valid_type=dict,
            non_db=True,
            required=False,
            help="Define options for the GW cacluation: walltime, memory, CPUs, etc.",
        )

        spec.input(
            "debug",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            required=False,
            help="Run with fast parameters for debugging.",
        )

        spec.outline(
            cls.setup,
            engine.while_(cls.scf_is_not_done)(cls.submit_scf, cls.check_scf),
            cls.submit_gw,
            cls.finalize,
        )
        spec.outputs.dynamic = True

        spec.exit_code(
            381,
            "ERROR_CONVERGENCE1",
            message="SCF of the first step did not converge.",
        )
        spec.exit_code(
            382,
            "ERROR_CONVERGENCE2",
            message="SCF of the second step did not converge.",
        )
        spec.exit_code(
            383,
            "ERROR_NEGATIVE_GAP",
            message="SCF produced a negative gap.",
        )
        spec.exit_code(
            390,
            "ERROR_TERMINATION",
            message="One or more steps of the work chain failed.",
        )

    def setup(self):
        self.report("Inspecting input and setting up things")

        if self.inputs.protocol not in ALLOWED_PROTOCOLS:
            self.report("Error: protocol not supported.")
            return self.exit_codes.ERROR_TERMINATION

        self.ctx.protocols = cp2k_utils.load_cp2k_protocol("gw_protocols.yml")
        structure = self.inputs.structure
        self.ctx.cutoff = cp2k_utils.get_cutoff(structure=structure)
        magnetization_per_site = copy.deepcopy(self.inputs.magnetization_per_site)
        ghost_per_site = None

        # Add ghost atoms in case of gw-ic.
        if self.inputs.run_image_charge:
            atoms = self.inputs.structure.get_ase()
            image = atoms.copy()
            image.positions[:, 2] = (
                2 * self.inputs.z_ic_plane.value - atoms.positions[:, 2]
            )
            ghost_per_site = [0 for a in atoms] + [1 for a in image]
            if magnetization_per_site:
                magnetization_per_site += [0 for i in range(len(image))]
            structure = orm.StructureData(ase=atoms + image)

        structure_with_tags, kinds_dict = cp2k_utils.determine_kinds(
            structure, magnetization_per_site, ghost_per_site
        )

        # KINDS section.
        self.ctx.kinds_section = cp2k_utils.get_kinds_section(
            kinds_dict, protocol=self.inputs.protocol
        )

        # Make sure cell is big enough for MT poisson solver.
        if self.inputs.debug:
            extra_cell = 5.0
        else:
            extra_cell = 15.0
        atoms = structure_with_tags.get_ase()
        atoms.cell = 2 * (np.ptp(atoms.positions, axis=0)) + extra_cell
        atoms.center()
        self.ctx.structure = orm.StructureData(ase=atoms)

        # Determine which basis and pseudo files to include
        if self.inputs.protocol in ["gapw_std", "gapw_hq"]:
            basis = "GW_BASIS_SET"
            potential = "ALL_POTENTIALS"
        elif self.inputs.protocol == "gpw_std":
            basis = "K_GW_BASIS"
            potential = "POTENTIAL"

        self.ctx.files = {
            "basis": orm.SinglefileData(
                file=pathlib.Path(__file__).parent / "data" / basis,
            ),
            "pseudo": orm.SinglefileData(
                file=pathlib.Path(__file__).parent / "data" / potential,
            ),
        }

        self.ctx.current_scf_protocol = None
        self.ctx.scf_restart_from_last = False

        return engine.ExitCode(0)

    def scf_is_not_done(self):
        if hasattr(self.ctx, "scf"):
            scf_out_params = self.ctx.scf.outputs.output_parameters

            scf_converged = scf_out_params["motion_step_info"]["scf_converged"][-1]

            if not scf_converged:
                self.report("SCF did not converge, try the next protocol.")
                self.ctx.scf_restart_from_last = False
                return True

            gap_positive = (
                min(
                    scf_out_params["bandgap_spin1_au"],
                    scf_out_params["bandgap_spin2_au"],
                )
                >= 0.0
            )

            if not gap_positive:
                self.report("Gap is negative, try the next protocol.")
                # If the SCF converged but the gap was negative, restart in the next step
                self.ctx.scf_restart_from_last = True
                return True

            self.report("SCF finished well, continue to GW.")
            return False

        return True

    def _check_and_set_uks(self, input_dict):
        if self.inputs.multiplicity.value > 0:
            input_dict["FORCE_EVAL"]["DFT"]["UKS"] = ".TRUE."
            input_dict["FORCE_EVAL"]["DFT"][
                "MULTIPLICITY"
            ] = self.inputs.multiplicity.value

    def _set_debug(self, input_dict):
        input_dict["FORCE_EVAL"]["DFT"]["PRINT"]["MO_CUBES"]["STRIDE"] = "6 6 6"
        input_dict["FORCE_EVAL"]["DFT"]["PRINT"]["E_DENSITY_CUBE"]["STRIDE"] = "6 6 6"
        input_dict["FORCE_EVAL"]["DFT"]["SCF"]["EPS_SCF"] = 0.2
        input_dict["FORCE_EVAL"]["DFT"]["SCF"]["EPS_EIGVAL"] = 0.2
        input_dict["FORCE_EVAL"]["DFT"]["SCF"]["OUTER_SCF"]["EPS_SCF"] = 0.2

    def submit_scf(self):
        # Try the next SCF section:
        if self.ctx.current_scf_protocol is None:
            # First try
            self.ctx.current_scf_protocol = "scf_ot_cg"
        elif self.ctx.current_scf_protocol == "scf_ot_cg":
            # Second try
            self.ctx.current_scf_protocol = "scf_ot_diis"
        elif self.ctx.current_scf_protocol == "scf_ot_diis":
            # Third try
            self.ctx.current_scf_protocol = "scf_diag_smearing"
        else:
            # Failure
            return self.exit_codes.ERROR_CONVERGENCE1

        self.report(f"Submitting SCF (protocol {self.ctx.current_scf_protocol})")

        # Build the input dictionary.
        step_protocol = self.inputs.protocol.value + "_scf_step"
        input_dict = copy.deepcopy(self.ctx.protocols[step_protocol])

        scf_section = copy.deepcopy(self.ctx.protocols[self.ctx.current_scf_protocol])
        input_dict["FORCE_EVAL"]["DFT"]["SCF"] = scf_section

        self._check_and_set_uks(input_dict)

        cp2k_utils.dict_merge(input_dict, self.ctx.kinds_section)

        input_dict["FORCE_EVAL"]["DFT"]["MGRID"]["CUTOFF"] = self.ctx.cutoff

        if self.inputs.debug:
            self._set_debug(input_dict)

        # Prepare the builder.
        builder = Cp2kCalculation.get_builder()
        builder.code = self.inputs.code
        builder.structure = self.ctx.structure
        builder.file = self.ctx.files

        if hasattr(self.ctx, "scf") and self.ctx.scf_restart_from_last:
            builder.parent_calc_folder = self.ctx.scf.outputs.remote_folder
            input_dict["FORCE_EVAL"]["DFT"][
                "RESTART_FILE_NAME"
            ] = "./parent_calc/aiida-RESTART.wfn"

        # Options.
        builder.metadata.options = self.inputs.options.scf
        if "max_wallclock_seconds" in self.inputs.options:
            input_dict["GLOBAL"]["WALLTIME"] = max(
                self.inputs.options["max_wallclock_seconds"] - 600, 600
            )
        builder.metadata.options["parser_name"] = "cp2k_advanced_parser"

        builder.parameters = orm.Dict(input_dict)

        return engine.ToContext(scf=self.submit(builder))

    def check_scf(self):
        return (
            engine.ExitCode(0)
            if self.ctx.scf.is_finished_ok
            else self.exit_codes.ERROR_TERMINATION
        )

    def submit_gw(self):
        self.report("Submitting GW.")

        # Build the input dictionary

        if self.inputs.run_image_charge:
            step_protocol = self.inputs.protocol.value + "_ic_step"
        else:
            step_protocol = self.inputs.protocol.value + "_gw_step"

        input_dict = copy.deepcopy(self.ctx.protocols[step_protocol])

        scf_section = copy.deepcopy(self.ctx.protocols[self.ctx.current_scf_protocol])
        input_dict["FORCE_EVAL"]["DFT"]["SCF"] = scf_section

        self._check_and_set_uks(input_dict)

        cp2k_utils.dict_merge(input_dict, self.ctx.kinds_section)

        input_dict["FORCE_EVAL"]["DFT"]["MGRID"]["CUTOFF"] = self.ctx.cutoff

        if self.inputs.debug:
            self._set_debug(input_dict)

        # Prepare the builder.
        builder = Cp2kCalculation.get_builder()
        builder.code = self.inputs.code
        builder.structure = self.ctx.structure
        builder.file = self.ctx.files

        # Restart from the wavefunction of the SCF obtained previously.
        builder.parent_calc_folder = self.ctx.scf.outputs.remote_folder

        # Options.
        builder.metadata.options = self.inputs.options.gw
        if "max_wallclock_seconds" in self.inputs.options:
            input_dict["GLOBAL"]["WALLTIME"] = max(
                self.inputs.options["max_wallclock_seconds"] - 600, 600
            )

        builder.metadata.options["parser_name"] = "nanotech_empa.cp2k_gw_parser"

        builder.parameters = orm.Dict(input_dict)

        return engine.ToContext(second_step=self.submit(builder))

    def finalize(self):
        self.report("Finalizing...")

        if not self.ctx.second_step.is_finished_ok:
            return self.exit_codes.ERROR_TERMINATION
        if not self.ctx.second_step.outputs.std_output_parameters["motion_step_info"][
            "scf_converged"
        ][-1]:
            self.report("GW step did not converge")
            return self.exit_codes.ERROR_CONVERGENCE2

        self.out(
            "std_output_parameters", self.ctx.second_step.outputs.std_output_parameters
        )
        self.out(
            "gw_output_parameters", self.ctx.second_step.outputs.gw_output_parameters
        )

        return engine.ExitCode(0)
