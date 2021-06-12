from aiida_nanotech_empa.utils import common_utils

from aiida.engine import WorkChain, ToContext, ExitCode
from aiida.orm import Int, Str, Code, Bool, Dict, StructureData

from aiida.plugins import WorkflowFactory

GaussianRelaxWorkChain = WorkflowFactory('nanotech_empa.gaussian.relax')
GaussianScfWorkChain = WorkflowFactory('nanotech_empa.gaussian.scf')


class GaussianRelaxScfCubesWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("gaussian_code", valid_type=Code)
        spec.input("formchk_code", valid_type=Code)
        spec.input("cubegen_code", valid_type=Code)

        spec.input('structure',
                   valid_type=StructureData,
                   required=True,
                   help='input geometry')

        spec.input('functional',
                   valid_type=Str,
                   required=True,
                   help='xc functional')

        spec.input('basis_set_opt',
                   valid_type=Str,
                   required=True,
                   help='basis_set for opt')

        spec.input('basis_set_scf',
                   valid_type=Str,
                   required=True,
                   help='basis_set for scf')

        spec.input('multiplicity',
                   valid_type=Int,
                   required=False,
                   default=lambda: Int(0),
                   help='spin multiplicity; 0 means RKS')

        spec.input(
            'options',
            valid_type=Dict,
            required=False,
            help="Use custom metadata.options instead of the automatic ones.")

        spec.outline(cls.relax, cls.scf_and_cubes, cls.finalize)

        spec.outputs.dynamic = True

        spec.exit_code(
            390,
            "ERROR_TERMINATION",
            message="One or more steps of the work chain failed.",
        )

    def is_uks(self):
        if self.inputs.multiplicity.value == 0:
            return False
        return True

    def relax(self):

        self.report("Submitting optimization")

        builder = GaussianRelaxWorkChain.get_builder()
        builder.gaussian_code = self.inputs.gaussian_code
        builder.structure = self.inputs.structure
        builder.functional = self.inputs.functional
        builder.basis_set = self.inputs.basis_set_opt

        builder.multiplicity = self.inputs.multiplicity
        builder.wfn_stable_opt = Bool(self.is_uks())

        if 'options' in self.inputs:
            builder.options = self.inputs.options

        submitted_node = self.submit(builder)
        return ToContext(opt=submitted_node)

    def scf_and_cubes(self):

        if not common_utils.check_if_calc_ok(self, self.ctx.opt):
            return self.exit_codes.ERROR_TERMINATION

        self.report("Submitting SCF and Cubes workchain")

        builder = GaussianScfWorkChain.get_builder()

        builder.gaussian_code = self.inputs.gaussian_code
        builder.formchk_code = self.inputs.formchk_code
        builder.cubegen_code = self.inputs.cubegen_code

        builder.structure = self.ctx.opt.outputs.output_structure
        builder.functional = self.inputs.functional
        builder.basis_set = self.inputs.basis_set_scf
        builder.multiplicity = self.inputs.multiplicity

        builder.parent_calc_folder = self.ctx.opt.outputs.remote_folder

        builder.n_occ = Int(1)
        builder.n_virt = Int(1)
        builder.cubegen_parser_params = Dict(dict={
            'heights': [3.0],
            'orient_cube': True,
            'isovalues': [0.01],
        })

        if 'options' in self.inputs:
            builder.options = self.inputs.options

        submitted_node = self.submit(builder)
        return ToContext(scf_and_cubes=submitted_node)

    def finalize(self):

        if not common_utils.check_if_calc_ok(self, self.ctx.scf_and_cubes):
            return self.exit_codes.ERROR_TERMINATION

        self.report("Finalizing...")

        self.out("opt_structure", self.ctx.opt.outputs.output_structure)
        self.out("opt_energy", self.ctx.opt.outputs.energy_ev)
        self.out("scf_energy", self.ctx.scf_and_cubes.outputs.energy_ev)
        self.out("scf_out_params",
                 self.ctx.scf_and_cubes.outputs.output_parameters)

        self.out("cube_image_folder",
                 self.ctx.scf_and_cubes.outputs.cube_image_folder)
        self.out("cube_planes_array",
                 self.ctx.scf_and_cubes.outputs.cube_planes_array)

        self.out("remote_folder", self.ctx.scf_and_cubes.outputs.remote_folder)

        return ExitCode(0)
