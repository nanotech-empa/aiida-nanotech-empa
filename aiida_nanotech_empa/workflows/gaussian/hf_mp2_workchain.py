from aiida import engine, orm, plugins

from ...utils import common_utils
from .scf_workchain import GaussianScfWorkChain

GaussianBaseWorkChain = plugins.WorkflowFactory("gaussian.base")
GaussianCubesWorkChain = plugins.WorkflowFactory("gaussian.cubes")


class GaussianHfMp2WorkChain(engine.WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("gaussian_code", valid_type=orm.Code)

        spec.input(
            "structure",
            valid_type=orm.StructureData,
            required=True,
            help="input geometry",
        )

        spec.input("basis_set", valid_type=orm.Str, required=True, help="basis_set")

        spec.input(
            "multiplicity",
            valid_type=orm.Int,
            required=False,
            default=lambda: orm.Int(1),
            help="spin multiplicity; 0 means RKS",
        )

        spec.input(
            "parent_calc_folder",
            valid_type=orm.RemoteData,
            required=False,
            help="the folder of a completed gaussian calculation",
        )

        # CUBE GENERATION INPUTS
        spec.input(
            "num_orbital_cubes",
            valid_type=orm.Int,
            required=False,
            default=lambda: orm.Int(0),
            help="Generate cubes for the mp2 orbitals (n*occ and n*virt).",
        )

        spec.input("formchk_code", valid_type=orm.Code, required=False)
        spec.input("cubegen_code", valid_type=orm.Code, required=False)

        spec.input(
            "edge_space",
            valid_type=orm.Float,
            required=False,
            default=lambda: orm.Float(3.0),
            help="Extra cube space in addition to molecule bounding box [ang].",
        )
        spec.input(
            "cubegen_parser_params",
            valid_type=orm.Dict,
            required=False,
            default=lambda: orm.Dict(dict={}),
            help="Additional parameters to cubegen parser.",
        )

        spec.input(
            "options",
            valid_type=orm.Dict,
            required=False,
            help="Use custom metadata.options instead of the automatic ones.",
        )

        spec.outline(cls.hf, cls.mp2, cls.finalize)

        spec.outputs.dynamic = True

        spec.exit_code(
            301,
            "ERROR_MULTIPLICITY",
            message="Multiplicity and number of el. doesn't match.",
        )
        spec.exit_code(
            302,
            "ERROR_OPTIONS",
            message="Input options are invalid.",
        )
        spec.exit_code(
            390,
            "ERROR_TERMINATION",
            message="One or more steps of the work chain failed.",
        )

    def hf(self):
        self.report("Submitting HF")

        builder = GaussianScfWorkChain.get_builder()
        builder.gaussian_code = self.inputs.gaussian_code

        builder.structure = self.inputs.structure
        builder.functional = orm.Str("hf")
        builder.basis_set = self.inputs.basis_set
        builder.multiplicity = self.inputs.multiplicity

        builder.wfn_stable_opt_min_basis = orm.Bool(True)

        if "options" in self.inputs:
            builder.options = self.inputs.options

        future = self.submit(builder)
        return engine.ToContext(hf=future)

    def should_do_cubes(self):
        codes_set = "formchk_code" in self.inputs and "cubegen_code" in self.inputs
        non_zero_num = self.inputs.num_orbital_cubes.value > 0
        return codes_set and non_zero_num

    def mp2(self):
        if not common_utils.check_if_calc_ok(self, self.ctx.hf):
            return self.exit_codes.ERROR_TERMINATION

        self.report("Submitting MP2")

        builder = GaussianScfWorkChain.get_builder()
        builder.gaussian_code = self.inputs.gaussian_code

        builder.structure = self.inputs.structure
        builder.functional = orm.Str("mp2")
        builder.basis_set = self.inputs.basis_set
        builder.multiplicity = self.inputs.multiplicity

        builder.parent_calc_folder = self.ctx.hf.outputs.remote_folder

        if self.should_do_cubes():
            builder.n_occ = self.inputs.num_orbital_cubes
            builder.n_virt = self.inputs.num_orbital_cubes
            builder.formchk_code = self.inputs.formchk_code
            builder.cubegen_code = self.inputs.cubegen_code
            builder.edge_space = self.inputs.edge_space
            builder.cubegen_parser_params = self.inputs.cubegen_parser_params

        if "options" in self.inputs:
            builder.options = self.inputs.options

        future = self.submit(builder)
        return engine.ToContext(mp2=future)

    def finalize(self):
        if not common_utils.check_if_calc_ok(self, self.ctx.mp2):
            return self.exit_codes.ERROR_TERMINATION

        self.report("Finalizing...")

        self.out("hf_output_parameters", self.ctx.hf.outputs.output_parameters)
        self.out("mp2_output_parameters", self.ctx.mp2.outputs.output_parameters)

        if self.should_do_cubes():
            self.out("mp2_cube_images", self.ctx.mp2.outputs.cube_image_folder)

        self.out("remote_folder", self.ctx.mp2.outputs.remote_folder)

        return engine.ExitCode(0)
