from aiida import engine, orm, plugins

from ...utils import common_utils
from .diag_workchain import Cp2kDiagWorkChain

BaderCalculation = plugins.CalculationFactory("nanotech_empa.bader")
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
        spec.outline(
            cls.setup,
            cls.run_ot_scf,
            engine.if_(cls.should_run_diag_scf)(
                cls.run_diag_scf,
                engine.if_(cls.should_run_sparse_overlap)(
                    cls.run_sparse_overlap,
                ),
            ),
            engine.if_(cls.should_run_bader)(
                cls.run_bader,
            ),
            cls.finalize,
        )
        spec.exit_code(
            391,
            "ERROR_MISSING_SPARSE_OVERLAP_CODE",
            message="A sparse overlap code is required to retrieve sparse overlap entries.",
        )
        spec.exit_code(
            392,
            "ERROR_MISSING_BADER_CODE",
            message="A Bader code is required to compute Bader charges.",
        )

    def should_run_diag_scf(self):
        if self.should_run_bader():
            return False

        dft_params = self.inputs.dft_params.get_dict()
        return (
            self.inputs.write_overlap_matrix.value
            or self.inputs.retrieve_sparse_overlap.value
            or dft_params.get("added_mos", 0) > 0
        )

    def should_run_bader(self):
        return self.inputs.compute_bader_charges.value

    def should_run_sparse_overlap(self):
        return self.inputs.retrieve_sparse_overlap.value

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
        if "sparse_overlap_code" not in self.inputs:
            return self.exit_codes.ERROR_MISSING_SPARSE_OVERLAP_CODE

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

    def run_bader(self):
        if "bader_code" not in self.inputs:
            return self.exit_codes.ERROR_MISSING_BADER_CODE

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
            self.out("bader_retrieved", self.ctx.bader.outputs.retrieved)
            self.report("Work chain is finished")
            return None

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
