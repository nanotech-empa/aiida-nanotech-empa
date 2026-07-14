from aiida import engine, orm, plugins

from ...utils import common_utils
from .diag_workchain import Cp2kDiagWorkChain

SparseOverlapCalculation = plugins.CalculationFactory("nanotech_empa.sparse_overlap")


class Cp2kScfWorkChain(Cp2kDiagWorkChain):
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
        spec.outline(
            cls.setup,
            cls.run_ot_scf,
            engine.if_(cls.should_run_diag_scf)(
                cls.run_diag_scf,
                engine.if_(cls.should_run_sparse_overlap)(
                    cls.run_sparse_overlap,
                ),
            ),
            cls.finalize,
        )
        spec.inputs.validator = staticmethod(cls._validate_inputs)

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

    def should_run_diag_scf(self):
        dft_params = self.inputs.dft_params.get_dict()
        return (
            self.inputs.write_overlap_matrix.value
            or self.inputs.retrieve_sparse_overlap.value
            or dft_params.get("added_mos", 0) > 0
        )

    def should_run_sparse_overlap(self):
        return self.inputs.retrieve_sparse_overlap.value

    def update_diag_input_dict(self, input_dict):
        if not (
            self.inputs.write_overlap_matrix.value
            or self.inputs.retrieve_sparse_overlap.value
        ):
            return

        print_section = input_dict["FORCE_EVAL"]["DFT"].setdefault("PRINT", {})
        print_section["AO_MATRICES"] = {
            "_": "ON",
            "OVERLAP": "T",
            "FILENAME": "overlap_matrix.out",
            "NDIGITS": self.inputs.overlap_ndigits.value,
        }

    def run_sparse_overlap(self):
        self.report("Running sparse overlap post-processing")
        builder = SparseOverlapCalculation.get_builder()
        builder.code = self.inputs.sparse_overlap_code
        builder.parent_calc_folder = self.ctx.diag_scf.outputs.remote_folder
        builder.threshold = self.inputs.overlap_threshold
        builder.matrix_filename = orm.Str("aiida-overlap_matrix.out-1_0.Log")
        builder.output_filename = orm.Str("sparse_overlap.npz")
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

    def finalize(self):
        if self.should_run_diag_scf():
            if not common_utils.check_if_calc_ok(self, self.ctx.diag_scf):
                self.report("diagonalization scf failed")
                return self.exit_codes.ERROR_TERMINATION

            if self.should_run_sparse_overlap():
                if not common_utils.check_if_calc_ok(self, self.ctx.sparse_overlap):
                    self.report("sparse overlap post-processing failed")
                    return self.exit_codes.ERROR_TERMINATION
                self.out(
                    "sparse_overlap_retrieved",
                    self.ctx.sparse_overlap.outputs.retrieved,
                )

            self.out("output_parameters", self.ctx.diag_scf.outputs.output_parameters)
            self.out("remote_folder", self.ctx.diag_scf.outputs.remote_folder)
            self.out("retrieved", self.ctx.diag_scf.outputs.retrieved)
            self.report("Work chain is finished")
            return None

        if not common_utils.check_if_calc_ok(self, self.ctx.ot_scf):
            self.report("OT SCF failed")
            return self.exit_codes.ERROR_TERMINATION

        self.out("output_parameters", self.ctx.ot_scf.outputs.output_parameters)
        self.out("remote_folder", self.ctx.ot_scf.outputs.remote_folder)
        self.out("retrieved", self.ctx.ot_scf.outputs.retrieved)
        self.report("Work chain is finished")
        return None
