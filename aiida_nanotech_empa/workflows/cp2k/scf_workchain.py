from aiida import engine, orm, plugins

from ...utils import common_utils
from .diag_workchain import Cp2kDiagWorkChain

BaderCalculation = plugins.CalculationFactory("nanotech_empa.bader")
SparseOverlapCalculation = plugins.CalculationFactory("nanotech_empa.sparse_overlap")
Cp2kUnfoldingCalculation = plugins.CalculationFactory("nanotech_empa.cp2k_unfolding")


class Cp2kScfWorkChain(Cp2kDiagWorkChain):
    """Extends `Cp2kDiagWorkChain`, making the diagonalization step optional
    and adding the option to print/retrieve the AO overlap matrix.
    """

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input(
            "write_overlap_matrix",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            required=False,
            help="Run the diagonalization SCF step and print the AO overlap matrix.",
        )
        spec.input(
            "retrieve_sparse_overlap",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            required=False,
            help="Post-process the remote AO overlap matrix and retrieve sparse entries.",
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
            "compute_bader_charges",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            required=False,
            help="Run Bader charge analysis on the OT charge-density cube.",
        )
        spec.input(
            "bader_code",
            valid_type=orm.Code,
            required=False,
            help="Bader executable configured for the nanotech_empa.bader plugin.",
        )
        spec.input(
            "bader_cutoff",
            valid_type=orm.Float,
            default=lambda: orm.Float(1200.0),
            required=False,
            help="Plane-wave cutoff used for the OT charge-density cube for Bader analysis.",
        )
        spec.input(
            "compute_unfolding",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            required=False,
            help="Post-process WFN and sparse AO overlap into unfolded band weights.",
        )
        spec.input(
            "unfolding_code",
            valid_type=orm.Code,
            required=False,
            help="Python code configured for the nanotech_empa.cp2k_unfolding plugin.",
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
            default=lambda: orm.Str("G-K-M-G"),
            required=False,
            help="High-symmetry path labels, e.g. G-K-M-G.",
        )
        spec.input(
            "unfolding_lattice_type",
            valid_type=orm.Str,
            default=lambda: orm.Str("auto"),
            required=False,
            help="1d, square, rectangular, hexagonal, oblique, or auto.",
        )
        spec.outline(
            cls.setup,
            cls.run_ot_scf,
            engine.if_(cls.should_run_diag_scf)(
                cls.run_diag_scf,
                engine.if_(cls.should_run_sparse_overlap)(
                    cls.run_sparse_overlap,
                ),
                engine.if_(cls.should_run_unfolding)(
                    cls.run_unfolding,
                ),
            ),
            engine.if_(cls.should_run_bader)(
                cls.run_bader,
            ),
            cls.finalize,
        )
        spec.inputs.validator = cls._validate_inputs
        spec.exit_code(
            395,
            "ERROR_MISSING_UNFOLDING_OUTPUT",
            message="CP2K band unfolding finished without retrieving unfolding_bands.npz.",
        )

    @staticmethod
    def _validate_inputs(value, port_namespace):
        if (
            value["retrieve_sparse_overlap"].value
            and "sparse_overlap_code" not in value
        ):
            return (
                "'sparse_overlap_code' is required when "
                "'retrieve_sparse_overlap' is True."
            )
        compute_bader_charges = value.get("compute_bader_charges")
        if (
            compute_bader_charges is not None
            and compute_bader_charges.value
            and "bader_code" not in value
        ):
            return "'bader_code' is required when 'compute_bader_charges' is True."
        compute_unfolding = value.get("compute_unfolding")
        if compute_unfolding is not None and compute_unfolding.value:
            if "unfolding_code" not in value:
                return "'unfolding_code' is required when 'compute_unfolding' is True."
            if "unfolding_primitive_vectors" not in value:
                return (
                    "'unfolding_primitive_vectors' is required when "
                    "'compute_unfolding' is True."
                )

    def should_run_diag_scf(self):
        if self.should_run_bader():
            return False

        dft_params = self.inputs.dft_params.get_dict()
        return (
            self.inputs.write_overlap_matrix.value
            or self.inputs.retrieve_sparse_overlap.value
            or self.inputs.compute_unfolding.value
            or dft_params.get("added_mos", 0) > 0
        )

    def should_run_bader(self):
        return self.inputs.compute_bader_charges.value

    def should_run_sparse_overlap(self):
        return self.inputs.retrieve_sparse_overlap.value

    def should_run_unfolding(self):
        return self.inputs.compute_unfolding.value

    def update_ot_input_dict(self, input_dict):
        if not self.should_run_bader():
            return

        input_dict["FORCE_EVAL"]["DFT"]["MGRID"]["CUTOFF"] = (
            self.inputs.bader_cutoff.value
        )
        print_section = input_dict["FORCE_EVAL"]["DFT"].setdefault("PRINT", {})
        charge_density = print_section.setdefault("E_DENSITY_CUBE", {})
        charge_density["STRIDE"] = "1 1 1"
        charge_density.setdefault("EACH", {"QS_SCF": "0", "GEO_OPT": "0"})
        charge_density.setdefault("ADD_LAST", "NUMERIC")

    def update_diag_input_dict(self, input_dict):
        if not (
            self.inputs.write_overlap_matrix.value
            or self.inputs.retrieve_sparse_overlap.value
            or self.inputs.compute_unfolding.value
        ):
            return

        print_section = input_dict["FORCE_EVAL"]["DFT"].setdefault("PRINT", {})
        print_section["AO_MATRICES"] = {
            "_": "ON",
            "OVERLAP": "T",
            # CP2K/aiida-cp2k retrieve this as "aiida-overlap_matrix.out-1_0.Log",
            # matching SparseOverlapCalculation's default matrix_filename.
            "FILENAME": "overlap_matrix.out",
            "NDIGITS": self.inputs.overlap_ndigits.value,
        }

    def run_sparse_overlap(self):
        self.report("Running sparse overlap post-processing")
        builder = SparseOverlapCalculation.get_builder()
        builder.code = self.inputs.sparse_overlap_code
        builder.parent_calc_folder = self.ctx.diag_scf.outputs.remote_folder
        builder.threshold = self.inputs.overlap_threshold
        builder.metadata = {
            "label": "sparse_overlap",
            "options": {
                "resources": {
                    "num_machines": 1,
                    "num_mpiprocs_per_machine": 1,
                    "num_cores_per_mpiproc": 1,
                },
                "max_wallclock_seconds": min(
                    3600, self.ctx.options["max_wallclock_seconds"]
                ),
                "withmpi": False,
            },
        }
        return engine.ToContext(sparse_overlap=self.submit(builder))

    def run_unfolding(self):
        self.report("Running CP2K band unfolding post-processing")
        builder = Cp2kUnfoldingCalculation.get_builder()
        builder.code = self.inputs.unfolding_code
        builder.parent_calc_folder = self.ctx.diag_scf.outputs.remote_folder
        builder.primitive_vectors = self.inputs.unfolding_primitive_vectors
        builder.path = self.inputs.unfolding_path
        builder.lattice_type = self.inputs.unfolding_lattice_type
        builder.overlap_threshold = self.inputs.overlap_threshold
        builder.metadata = {
            "label": "cp2k_unfolding",
            "options": {
                "resources": {
                    "num_machines": 1,
                    "num_mpiprocs_per_machine": 1,
                    "num_cores_per_mpiproc": 1,
                },
                "max_wallclock_seconds": min(
                    7200, self.ctx.options["max_wallclock_seconds"]
                ),
                "withmpi": False,
            },
        }
        return engine.ToContext(unfolding=self.submit(builder))

    def run_bader(self):
        if not common_utils.check_if_calc_ok(self, self.ctx.ot_scf):
            self.report("OT SCF failed")
            return self.exit_codes.ERROR_TERMINATION

        self.report("Running Bader charge analysis")
        builder = BaderCalculation.get_builder()
        builder.code = self.inputs.bader_code
        builder.parent_calc_folder = self.ctx.ot_scf.outputs.remote_folder
        builder.charge_density_filename = orm.Str("aiida-ELECTRON_DENSITY-1_0.cube")
        builder.metadata = {
            "label": "bader",
            "options": {
                "resources": {
                    "num_machines": 1,
                    "num_mpiprocs_per_machine": 1,
                    "num_cores_per_mpiproc": 1,
                },
                "max_wallclock_seconds": min(
                    3600, self.ctx.options["max_wallclock_seconds"]
                ),
                "withmpi": False,
            },
        }
        return engine.ToContext(bader=self.submit(builder))

    def finalize(self):
        if self.should_run_bader():
            if not common_utils.check_if_calc_ok(self, self.ctx.ot_scf):
                self.report("OT SCF failed")
                return self.exit_codes.ERROR_TERMINATION

            if not common_utils.check_if_calc_ok(self, self.ctx.bader):
                self.report("Bader charge analysis failed")
                return self.exit_codes.ERROR_TERMINATION

            self.out("output_parameters", self.ctx.ot_scf.outputs.output_parameters)
            self.out("remote_folder", self.ctx.ot_scf.outputs.remote_folder)
            self.out("retrieved", self.ctx.ot_scf.outputs.retrieved)
            self.out("ot_retrieved", self.ctx.ot_scf.outputs.retrieved)
            self.out("bader_retrieved", self.ctx.bader.outputs.retrieved)
            common_utils.add_extras(self.inputs.structure, "surfaces", self.node.uuid)
            self.report("Work chain is finished")
            return None

        run_diag = self.should_run_diag_scf()
        final_calc = self.ctx.diag_scf if run_diag else self.ctx.ot_scf
        calc_label = "diagonalization scf" if run_diag else "OT SCF"

        if not common_utils.check_if_calc_ok(self, final_calc):
            self.report(f"{calc_label} failed")
            return self.exit_codes.ERROR_TERMINATION

        if run_diag and self.should_run_sparse_overlap():
            if not common_utils.check_if_calc_ok(self, self.ctx.sparse_overlap):
                self.report("sparse overlap post-processing failed")
                return self.exit_codes.ERROR_TERMINATION
            self.out(
                "sparse_overlap_retrieved", self.ctx.sparse_overlap.outputs.retrieved
            )

        if run_diag and self.should_run_unfolding():
            if not common_utils.check_if_calc_ok(self, self.ctx.unfolding):
                self.report("CP2K band unfolding post-processing failed")
                return self.exit_codes.ERROR_TERMINATION
            output_filename = self.ctx.unfolding.inputs.output_filename.value
            retrieved_names = (
                self.ctx.unfolding.outputs.retrieved.base.repository.list_object_names()
            )
            if output_filename not in retrieved_names:
                self.report(f"CP2K band unfolding did not retrieve {output_filename}")
                return self.exit_codes.ERROR_MISSING_UNFOLDING_OUTPUT
            self.out("unfolding_retrieved", self.ctx.unfolding.outputs.retrieved)

        self.out("output_parameters", final_calc.outputs.output_parameters)
        self.out("remote_folder", final_calc.outputs.remote_folder)
        self.out("retrieved", final_calc.outputs.retrieved)
        self.out("ot_retrieved", self.ctx.ot_scf.outputs.retrieved)
        common_utils.add_extras(self.inputs.structure, "surfaces", self.node.uuid)
        self.report("Work chain is finished")
