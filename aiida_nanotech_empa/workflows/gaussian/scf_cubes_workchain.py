from aiida_nanotech_empa.workflows.gaussian import common

from aiida.engine import WorkChain, ToContext, ExitCode
from aiida.orm import Int, Str, Bool, Code, Dict, List
from aiida.orm import StructureData, RemoteData

from aiida.plugins import WorkflowFactory

GaussianBaseWorkChain = WorkflowFactory('gaussian.base')
GaussianCubesWorkChain = WorkflowFactory('gaussian.cubes')

# -------------------------------------------------------------
# Work Chain to run SCF and then CUBES
# -------------------------------------------------------------


class GaussianScfCubesWorkChain(WorkChain):
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

        spec.input('basis_set',
                   valid_type=Str,
                   required=True,
                   help='basis_set')

        spec.input('multiplicity',
                   valid_type=Int,
                   required=False,
                   default=lambda: Int(1),
                   help='spin multiplicity; 0 means RKS')

        spec.input('wfn_stable_opt',
                   valid_type=Bool,
                   required=False,
                   default=lambda: Bool(False),
                   help='if true, perform wfn stability optimization')

        spec.input('parent_calc_folder',
                   valid_type=RemoteData,
                   required=False,
                   help="the folder of a completed gaussian calculation")

        spec.input('n_occ',
                   valid_type=Int,
                   required=False,
                   default=lambda: Int(1),
                   help='Number of occupied orbital cubes to generate')

        spec.input('n_virt',
                   valid_type=Int,
                   required=False,
                   default=lambda: Int(1),
                   help='Number of virtual orbital cubes to generate')

        spec.input('isosurfaces',
                   valid_type=List,
                   required=False,
                   default=lambda: List(list=[0.010]),
                   help='Generated images isosurface isovalues')

        spec.input(
            'options',
            valid_type=Dict,
            required=False,
            help="Use custom metadata.options instead of the automatic ones.")

        spec.outline(cls.setup, cls.scf, cls.cubes, cls.finalize)

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

        common.setup_context_variables(self)

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

    def scf(self):
        self.report("Submitting SCF")

        parameters = Dict(
            dict={
                'link0_parameters': self.ctx.link0.copy(),
                'functional': self.ctx.functional,
                'basis_set': self.inputs.basis_set.value,
                'charge': 0,
                'multiplicity': self.ctx.mult,
                'route_parameters': {
                    'scf': {
                        'maxcycle': 128,
                        'conver': 7
                    },
                    'nosymm': None,
                },
            })
        if self.inputs.wfn_stable_opt:
            parameters['route_parameters']['stable'] = "opt"
        else:
            parameters['route_parameters']['sp'] = None

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
        return ToContext(scf=future)

    def cubes(self):

        if not common.check_if_previous_calc_ok(self, self.ctx.scf):
            return self.exit_codes.ERROR_TERMINATION

        self.report("Generating cubes")

        future = self.submit(
            GaussianCubesWorkChain,
            formchk_code=self.inputs.formchk_code,
            cubegen_code=self.inputs.cubegen_code,
            gaussian_calc_folder=self.ctx.scf.outputs.remote_folder,
            gaussian_output_params=self.ctx.scf.outputs['output_parameters'],
            n_occ=self.inputs.n_occ,
            n_virt=self.inputs.n_virt,
            cubegen_parser_name='nanotech_empa.gaussian.cubegen_pymol',
            cubegen_parser_params=Dict(
                dict={'isovalues': list(self.inputs.isosurfaces)}))
        return ToContext(cubes=future)

    def finalize(self):

        if not common.check_if_previous_calc_ok(self, self.ctx.cubes):
            return self.exit_codes.ERROR_TERMINATION

        self.report("Finalizing...")

        self.out("scf_energy", self.ctx.scf.outputs.energy_ev)
        self.out("scf_out_params", self.ctx.scf.outputs['output_parameters'])

        self.out("remote_folder", self.ctx.scf.outputs['remote_folder'])

        self.out("cube_image_folder",
                 self.ctx.cubes.outputs['cube_image_folder'])

        #for cubes_out in list(self.ctx.cubes.outputs):
        #    if cubes_out.startswith("cube"):
        #        self.out(cubes_out, self.ctx.cubes.outputs[cubes_out])
        #    elif cubes_out == 'retrieved':
        #        self.out("cubes_retrieved", self.ctx.cubes.outputs[cubes_out])

        return ExitCode(0)
