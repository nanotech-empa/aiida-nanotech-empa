from aiida_nanotech_empa.workflows.gaussian import common
from aiida_nanotech_empa.utils import common_utils

from aiida.engine import WorkChain, ToContext, ExitCode
from aiida.orm import Int, Str, Code, Dict, List
from aiida.orm import StructureData, RemoteData

from aiida.plugins import WorkflowFactory

GaussianBaseWorkChain = WorkflowFactory('gaussian.base')
GaussianCubesWorkChain = WorkflowFactory('gaussian.cubes')
GaussianScfCubesWorkChain = WorkflowFactory('nanotech_empa.gaussian.scf_cubes')

# -------------------------------------------------------------
# Work Chain to run HF and MP2
# -------------------------------------------------------------


class GaussianHfMp2WorkChain(WorkChain):
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

        spec.input('basis_set',
                   valid_type=Str,
                   required=True,
                   help='basis_set')

        spec.input('multiplicity',
                   valid_type=Int,
                   required=False,
                   default=lambda: Int(1),
                   help='spin multiplicity; 0 means RKS')

        spec.input('parent_calc_folder',
                   valid_type=RemoteData,
                   required=False,
                   help="the folder of a completed gaussian calculation")

        spec.input(
            'options',
            valid_type=Dict,
            required=False,
            help="Use custom metadata.options instead of the automatic ones.")

        spec.outline(cls.setup, cls.hf_small_basis_stable, cls.hf_production,
                     cls.mp2, cls.finalize)

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

    def is_uks(self):
        if self.inputs.multiplicity.value == 0:
            return False
        return True

    def setup(self):
        self.report("Inspecting input and setting up things")

        pymatgen_structure = self.inputs.structure.get_pymatgen_molecule()
        self.ctx.n_atoms = pymatgen_structure.num_sites
        self.ctx.n_electrons = pymatgen_structure.nelectrons

        if self.inputs.multiplicity.value == 0:
            self.ctx.mult = 1
        else:
            self.ctx.mult = self.inputs.multiplicity.value

        self.ctx.comp = self.inputs.gaussian_code.computer

        if self.ctx.mult % 2 == self.ctx.n_electrons % 2:
            return self.exit_codes.ERROR_MULTIPLICITY

        success = common.determine_metadata_options(self)
        if not success:
            return self.exit_codes.ERROR_OPTIONS

        num_cores, memory_mb = common.get_gaussian_cores_and_memory(
            self.ctx.metadata_options, self.ctx.comp)

        self.ctx.link0 = {
            '%chk': 'aiida.chk',
            '%mem': "%dMB" % memory_mb,
            '%nprocshared': str(num_cores),
        }

        return ExitCode(0)

    def hf_small_basis_stable(self):
        self.report("Submitting HF stability")

        parameters = Dict(
            dict={
                'link0_parameters': self.ctx.link0.copy(),
                'functional': 'uhf' if self.is_uks() else 'hf',
                'basis_set': 'STO-3G',
                'charge': 0,
                'multiplicity': self.ctx.mult,
                'route_parameters': {
                    'scf': {
                        'maxcycle': 128,
                        'conver': 7
                    },
                    'stable': 'opt',
                },
            })

        builder = GaussianBaseWorkChain.get_builder()

        if "parent_calc_folder" in self.inputs:
            # Read WFN from parent calc
            parameters['link0_parameters']['%oldchk'] = "parent_calc/aiida.chk"
            parameters['route_parameters']['guess'] = "read"
            builder.gaussian.parent_calc_folder = self.inputs.parent_calc_folder
        elif self.is_uks() and self.ctx.mult == 1:
            # For open-shell singlet, mix homo & lumo
            parameters['route_parameters']['guess'] = "mix"

        builder.gaussian.parameters = parameters
        builder.gaussian.structure = self.inputs.structure
        builder.gaussian.code = self.inputs.gaussian_code
        builder.gaussian.metadata.options = self.ctx.metadata_options

        future = self.submit(builder)
        return ToContext(hf_stable=future)

    def hf_production(self):

        if not common_utils.check_if_calc_ok(self, self.ctx.hf_stable):
            return self.exit_codes.ERROR_TERMINATION

        self.report("Submitting HF production run")

        builder = GaussianScfCubesWorkChain.get_builder()
        builder.gaussian_code = self.inputs.gaussian_code
        builder.formchk_code = self.inputs.formchk_code
        builder.cubegen_code = self.inputs.cubegen_code

        builder.structure = self.inputs.structure
        builder.functional = Str('hf')
        builder.basis_set = self.inputs.basis_set
        builder.multiplicity = self.inputs.multiplicity

        builder.parent_calc_folder = self.ctx.hf_stable.outputs.remote_folder
        if 'options' in self.inputs:
            builder.options = self.inputs.options

        builder.n_occ = Int(2)
        builder.n_virt = Int(2)
        builder.isosurfaces = List(list=[0.010, 0.005])

        future = self.submit(builder)
        return ToContext(hf=future)

    def mp2(self):

        if not common_utils.check_if_calc_ok(self, self.ctx.hf):
            return self.exit_codes.ERROR_TERMINATION

        self.report("Submitting MP2")

        parameters = Dict(
            dict={
                'link0_parameters': self.ctx.link0.copy(),
                'functional': 'ump2' if self.is_uks() else 'mp2',
                'basis_set': self.inputs.basis_set.value,
                'charge': 0,
                'multiplicity': self.ctx.mult,
                'route_parameters': {
                    'scf': {
                        'maxcycle': 128
                    },
                    'sp': None,
                },
            })

        builder = GaussianBaseWorkChain.get_builder()

        # Read WFN from the HF
        parameters['link0_parameters']['%oldchk'] = "parent_calc/aiida.chk"
        parameters['route_parameters']['guess'] = "read"
        builder.gaussian.parent_calc_folder = self.ctx.hf.outputs.remote_folder

        builder.gaussian.parameters = parameters
        builder.gaussian.structure = self.inputs.structure
        builder.gaussian.code = self.inputs.gaussian_code
        builder.gaussian.metadata.options = self.ctx.metadata_options

        future = self.submit(builder)
        return ToContext(mp2=future)

    def finalize(self):

        if not common_utils.check_if_calc_ok(self, self.ctx.mp2):
            return self.exit_codes.ERROR_TERMINATION

        self.report("Finalizing...")

        self.out("hf_output_parameters", self.ctx.hf.outputs.scf_out_params)
        self.out("mp2_output_parameters",
                 self.ctx.mp2.outputs.output_parameters)

        self.out("hf_cube_images", self.ctx.hf.outputs.cube_image_folder)

        self.out("remote_folder", self.ctx.mp2.outputs.remote_folder)

        return ExitCode(0)
