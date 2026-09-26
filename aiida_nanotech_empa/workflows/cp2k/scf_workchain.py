from aiida import engine, orm, plugins

from ...plugins import sparse_overlap, unfolding
from ...utils import common_utils
from .diag_workchain import Cp2kDiagWorkChain

BaderCalculation = plugins.CalculationFactory("nanotech_empa.bader")
SparseOverlapCalculation = plugins.CalculationFactory("nanotech_empa.sparse_overlap")
Cp2kUnfoldingCalculation = plugins.CalculationFactory("nanotech_empa.cp2k_unfolding")

OVERLAP_MATRIX_OPTIONS = ("none", "remote_only", "remote_and_sparse_retrieved")

# dft_params keys and inputs that only the diagonalization SCF step reads.
DIAG_ONLY_DFT_PARAMS = (
    "added_mos",
    "sc_diag",
    "smear_t",
    "nhomo",
    "nlumo",
    "elpa_switch",
)
DIAG_ONLY_INPUTS = ("settings", "pdos_lists")


def _print_overlap_matrix(input_dict, ndigits):
    print_section = input_dict["FORCE_EVAL"]["DFT"].setdefault("PRINT", {})
    print_section["AO_MATRICES"] = {
        "_": "ON",
        "OVERLAP": "T",
        "FILENAME": sparse_overlap.OVERLAP_FILENAME,
        "NDIGITS": ndigits,
    }


class Cp2kScfWorkChain(Cp2kDiagWorkChain):
    """Extends `Cp2kDiagWorkChain`, making the diagonalization step optional
    and adding the option to print/retrieve the AO overlap matrix and to run
    Bader charge analysis on the OT SCF charge density.
    """

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input(
            "run_diag_scf",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            required=False,
            help="Run the diagonalization SCF step after the OT SCF step.",
        )
        spec.input(
            "overlap_matrix",
            valid_type=orm.Str,
            default=lambda: orm.Str("none"),
            required=False,
            help="AO overlap matrix, printed in the last SCF step: 'none', "
            "'remote_only' (kept in its remote folder) or "
            "'remote_and_sparse_retrieved' (also converted to a sparse .npz "
            "and retrieved).",
        )
        spec.input(
            "sparse_overlap_code",
            valid_type=orm.Code,
            required=False,
            help="Python code configured for the nanotech_empa.sparse_overlap plugin.",
        )
        spec.input(
            "overlap_ndigits",
            valid_type=orm.Int,
            default=lambda: orm.Int(14),
            required=False,
            help="Number of digits for the printed AO overlap matrix.",
        )
        spec.input(
            "overlap_threshold",
            valid_type=orm.Float,
            default=lambda: orm.Float(1.0e-10),
            required=False,
            help="Absolute-value threshold for retrieved sparse overlap entries.",
        )
        spec.input(
            "bader_code",
            valid_type=orm.Code,
            required=False,
            help="Bader executable configured for the nanotech_empa.bader plugin. "
            "If given, the OT SCF prints its charge-density cube at a cutoff of "
            "at least 'bader_cutoff' and Bader charge analysis runs on it.",
        )
        spec.input(
            "bader_cutoff",
            valid_type=orm.Float,
            default=lambda: orm.Float(1200.0),
            required=False,
            help="Minimum plane-wave cutoff (Ry) of the OT SCF when Bader analysis "
            "runs. A higher cutoff from 'dft_params' or the structure is kept.",
        )
        spec.input(
            "unfolding_code",
            valid_type=orm.Code,
            required=False,
            help="Python code configured for the nanotech_empa.cp2k_unfolding plugin. "
            "If given, the diagonalization SCF wavefunction and AO overlap matrix "
            "are post-processed into unfolded band weights.",
        )
        spec.input(
            "unfolding_primitive_vectors",
            valid_type=orm.Str,
            required=False,
            help="Approximate primitive vectors as rows, separated by semicolons or newlines.",
        )
        spec.input(
            "unfolding_path",
            valid_type=orm.Str,
            default=lambda: orm.Str(unfolding.DEFAULT_PATH),
            required=False,
            help="High-symmetry path labels, e.g. G-K-M-G.",
        )
        spec.input(
            "unfolding_lattice_type",
            valid_type=orm.Str,
            default=lambda: orm.Str(unfolding.DEFAULT_LATTICE_TYPE),
            required=False,
            validator=unfolding.validate_lattice_type,
            help=f"One of {', '.join(unfolding.LATTICE_TYPES)}.",
        )
        spec.outline(
            cls.setup,
            cls.run_ot_scf,
            engine.if_(cls.should_run_diag_scf)(cls.run_diag_scf),
            engine.if_(cls.should_run_sparse_overlap)(cls.run_sparse_overlap),
            engine.if_(cls.should_run_unfolding)(cls.run_unfolding),
            engine.if_(cls.should_run_bader)(cls.run_bader),
            cls.finalize,
        )
        spec.inputs.validator = cls._validate_inputs

    @staticmethod
    def _validate_inputs(value, port_namespace):
        overlap_matrix = value["overlap_matrix"].value
        if overlap_matrix not in OVERLAP_MATRIX_OPTIONS:
            return (
                f"'overlap_matrix' must be one of {', '.join(OVERLAP_MATRIX_OPTIONS)}."
            )
        if (
            overlap_matrix == "remote_and_sparse_retrieved"
            and "sparse_overlap_code" not in value
        ):
            return (
                "'sparse_overlap_code' is required when "
                "'overlap_matrix' is 'remote_and_sparse_retrieved'."
            )
        if "unfolding_code" in value and (
            not value["run_diag_scf"].value or overlap_matrix == "none"
        ):
            return (
                "'unfolding_code' uses the diagonalization SCF wavefunction and "
                "AO overlap matrix: set 'run_diag_scf' and 'overlap_matrix'."
            )
        if "unfolding_code" in value and "unfolding_primitive_vectors" not in value:
            return "'unfolding_primitive_vectors' is required with 'unfolding_code'."
        if (
            "unfolding_code" in value
            and value["dft_params"].get("periodic", "XYZ") == "NONE"
        ):
            return (
                "'unfolding_code' requires a periodic system, not 'periodic': 'NONE'."
            )

        if "bader_code" in value and value["run_diag_scf"].value:
            return (
                "'bader_code' runs Bader on the OT SCF charge density and "
                "cannot be combined with 'run_diag_scf'."
            )

        if value["run_diag_scf"].value:
            return None

        ignored = [key for key in DIAG_ONLY_DFT_PARAMS if value["dft_params"].get(key)]
        ignored += [name for name in DIAG_ONLY_INPUTS if name in value]
        if ignored:
            return (
                f"{', '.join(ignored)} only affect the diagonalization SCF step, "
                "which is skipped. Set 'run_diag_scf' to run it."
            )

    def should_run_diag_scf(self):
        return self.inputs.run_diag_scf.value

    def should_run_bader(self):
        return "bader_code" in self.inputs

    def should_run_sparse_overlap(self):
        return self.inputs.overlap_matrix.value == "remote_and_sparse_retrieved"

    def should_run_unfolding(self):
        return "unfolding_code" in self.inputs

    def update_ot_input_dict(self, input_dict):
        # Bader reads the final OT density, printed on the full grid.
        if self.should_run_bader():
            mgrid = input_dict["FORCE_EVAL"]["DFT"]["MGRID"]
            if mgrid["CUTOFF"] < self.inputs.bader_cutoff.value:
                self.report(
                    f"Raising the OT SCF cutoff from {mgrid['CUTOFF']} to "
                    f"{self.inputs.bader_cutoff.value} Ry for Bader analysis"
                )
                mgrid["CUTOFF"] = self.inputs.bader_cutoff.value
            print_section = input_dict["FORCE_EVAL"]["DFT"].setdefault("PRINT", {})
            charge_density = print_section.setdefault("E_DENSITY_CUBE", {})
            charge_density["STRIDE"] = "1 1 1"
            charge_density.setdefault("EACH", {"QS_SCF": "0", "GEO_OPT": "0"})
            charge_density.setdefault("ADD_LAST", "NUMERIC")

        # The overlap matrix is printed in the last SCF step only.
        if (
            self.inputs.overlap_matrix.value != "none"
            and not self.inputs.run_diag_scf.value
        ):
            _print_overlap_matrix(input_dict, self.inputs.overlap_ndigits.value)

    def update_diag_input_dict(self, input_dict):
        if self.inputs.overlap_matrix.value != "none":
            _print_overlap_matrix(input_dict, self.inputs.overlap_ndigits.value)

    def _serial_postprocessing_metadata(self, label):
        """Single-core job, capped at one hour or the SCF wall time."""
        return {
            "label": label,
            "options": {
                "resources": {
                    "num_machines": 1,
                    "num_mpiprocs_per_machine": 1,
                    "num_cores_per_mpiproc": 1,
                },
                "max_wallclock_seconds": min(
                    3600, self.ctx.options["max_wallclock_seconds"]
                ),
            },
        }

    def run_sparse_overlap(self):
        final_calc = (
            self.ctx.diag_scf if self.should_run_diag_scf() else self.ctx.ot_scf
        )
        if not common_utils.check_if_calc_ok(self, final_calc):
            return self.exit_codes.ERROR_TERMINATION

        self.report("Running sparse overlap post-processing")
        builder = SparseOverlapCalculation.get_builder()
        builder.code = self.inputs.sparse_overlap_code
        builder.parent_calc_folder = final_calc.outputs.remote_folder
        builder.threshold = self.inputs.overlap_threshold
        builder.metadata = self._serial_postprocessing_metadata("sparse_overlap")
        return engine.ToContext(sparse_overlap=self.submit(builder))

    def run_unfolding(self):
        if not common_utils.check_if_calc_ok(self, self.ctx.diag_scf):
            return self.exit_codes.ERROR_TERMINATION

        self.report("Running CP2K band unfolding post-processing")
        builder = Cp2kUnfoldingCalculation.get_builder()
        builder.code = self.inputs.unfolding_code
        builder.parent_calc_folder = self.ctx.diag_scf.outputs.remote_folder
        builder.primitive_vectors = self.inputs.unfolding_primitive_vectors
        builder.path = self.inputs.unfolding_path
        builder.lattice_type = self.inputs.unfolding_lattice_type
        builder.overlap_threshold = self.inputs.overlap_threshold
        builder.metadata = self._serial_postprocessing_metadata("cp2k_unfolding")
        return engine.ToContext(unfolding=self.submit(builder))

    def run_bader(self):
        if not common_utils.check_if_calc_ok(self, self.ctx.ot_scf):
            self.report("OT SCF failed")
            return self.exit_codes.ERROR_TERMINATION

        self.report("Running Bader charge analysis")
        builder = BaderCalculation.get_builder()
        builder.code = self.inputs.bader_code
        builder.parent_calc_folder = self.ctx.ot_scf.outputs.remote_folder
        builder.metadata = self._serial_postprocessing_metadata("bader")
        return engine.ToContext(bader=self.submit(builder))

    def finalize(self):
        run_diag = self.should_run_diag_scf()
        final_calc = self.ctx.diag_scf if run_diag else self.ctx.ot_scf
        calc_label = "diagonalization scf" if run_diag else "OT SCF"

        if not common_utils.check_if_calc_ok(self, final_calc):
            self.report(f"{calc_label} failed")
            return self.exit_codes.ERROR_TERMINATION

        if self.should_run_sparse_overlap():
            if not common_utils.check_if_calc_ok(self, self.ctx.sparse_overlap):
                self.report("sparse overlap post-processing failed")
                return self.exit_codes.ERROR_TERMINATION
            self.out(
                "sparse_overlap_retrieved", self.ctx.sparse_overlap.outputs.retrieved
            )

        if self.should_run_bader():
            if not common_utils.check_if_calc_ok(self, self.ctx.bader):
                self.report("Bader charge analysis failed")
                return self.exit_codes.ERROR_TERMINATION
            self.out("bader_retrieved", self.ctx.bader.outputs.retrieved)

        if self.should_run_unfolding():
            if not common_utils.check_if_calc_ok(self, self.ctx.unfolding):
                self.report("CP2K band unfolding post-processing failed")
                return self.exit_codes.ERROR_TERMINATION
            self.out("unfolding_retrieved", self.ctx.unfolding.outputs.retrieved)

        self.out("output_parameters", final_calc.outputs.output_parameters)
        self.out("remote_folder", final_calc.outputs.remote_folder)
        self.out("retrieved", final_calc.outputs.retrieved)
        self.out("ot_retrieved", self.ctx.ot_scf.outputs.retrieved)
        common_utils.add_extras(self.inputs.structure, "surfaces", self.node.uuid)
        self.report("Work chain is finished")
