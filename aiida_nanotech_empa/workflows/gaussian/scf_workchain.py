from aiida_nanotech_empa.workflows.gaussian import common
from aiida_nanotech_empa.utils import common_utils

from aiida.engine import WorkChain, ToContext, ExitCode, if_
from aiida.orm import Int, Str, Bool, Code, Dict, Float, List
from aiida.orm import StructureData, RemoteData

from aiida.plugins import WorkflowFactory

GaussianBaseWorkChain = WorkflowFactory('gaussian.base')
GaussianCubesWorkChain = WorkflowFactory('gaussian.cubes')

# -------------------------------------------------------------
# Work Chain to run SCF and possibly CUBES
# -------------------------------------------------------------


class GaussianScfWorkChain(WorkChain):
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

        spec.input('wfn_stable_opt_min_basis',
                   valid_type=Bool,
                   required=False,
                   default=lambda: Bool(False),
                   help='if true, perform first a minimal basis stability opt')

        spec.input('parent_calc_folder',
                   valid_type=RemoteData,
                   required=False,
                   help="the folder of a completed gaussian calculation")

        # -------------------------------------------------------------------
        # CUBE GENERATION INPUTS
        spec.input('n_occ',
                   valid_type=Int,
                   required=False,
                   default=lambda: Int(0),
                   help='Number of occupied orbital cubes to generate')

        spec.input('n_virt',
                   valid_type=Int,
                   required=False,
                   default=lambda: Int(0),
                   help='Number of virtual orbital cubes to generate')

        spec.input("formchk_code", valid_type=Code, required=False)
        spec.input("cubegen_code", valid_type=Code, required=False)

        spec.input(
            'edge_space',
            valid_type=Float,
            required=False,
            default=lambda: Float(3.0),
            help='Extra cube space in addition to molecule bounding box [ang].'
        )
        spec.input(
            "cubegen_parser_name",
            valid_type=str,
            default='nanotech_empa.gaussian.cubegen_pymol',
            non_db=True,
        )
        spec.input("cubegen_parser_params",
                   valid_type=Dict,
                   required=False,
                   default=lambda: Dict(dict={}),
                   help='Additional parameters to cubegen parser.')
        # -------------------------------------------------------------------

        spec.input(
            'options',
            valid_type=Dict,
            required=False,
            help="Use custom metadata.options instead of the automatic ones.")

        spec.outline(
            cls.setup,
            if_(cls.should_do_min_basis_stable_opt)(cls.min_basis_stable_opt),
            cls.scf,
            if_(cls.should_do_cubes)(cls.cubes), cls.finalize)

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

    def should_do_min_basis_stable_opt(self):
        return self.inputs.wfn_stable_opt_min_basis

    def min_basis_stable_opt(self):
        self.report("Submitting minimal basis stable opt")

        parameters = Dict(
            dict={
                'link0_parameters': self.ctx.link0.copy(),
                'functional': self.ctx.functional,
                'basis_set': 'STO-3G',
                'charge': 0,
                'multiplicity': self.ctx.mult,
                'route_parameters': {
                    'scf': {
                        'maxcycle': 128
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
        return ToContext(min_stable=future)

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
                        'maxcycle': 128
                    },
                },
            })
        if self.inputs.wfn_stable_opt:
            parameters['route_parameters']['stable'] = "opt"
        else:
            parameters['route_parameters']['sp'] = None

        builder = GaussianBaseWorkChain.get_builder()

        if self.should_do_min_basis_stable_opt():
            # Read WFN from min basis stable opt
            parameters['link0_parameters']['%oldchk'] = "parent_calc/aiida.chk"
            parameters['route_parameters']['guess'] = "read"
            builder.gaussian.parent_calc_folder = self.ctx.min_stable.outputs.remote_folder
        elif "parent_calc_folder" in self.inputs:
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

    def should_do_cubes(self):
        codes_set = 'formchk_code' in self.inputs and 'cubegen_code' in self.inputs
        non_zero_num = self.inputs.n_occ.value > 0 and self.inputs.n_virt.value > 0
        return codes_set and non_zero_num

    def cubes(self):

        if not common_utils.check_if_calc_ok(self, self.ctx.scf):
            return self.exit_codes.ERROR_TERMINATION

        self.report("Generating cubes")

        orb_index_list = list(
            range(-self.inputs.n_occ.value + 1, self.inputs.n_virt.value + 1))

        future = self.submit(
            GaussianCubesWorkChain,
            formchk_code=self.inputs.formchk_code,
            cubegen_code=self.inputs.cubegen_code,
            gaussian_calc_folder=self.ctx.scf.outputs.remote_folder,
            gaussian_output_params=self.ctx.scf.outputs['output_parameters'],
            orbital_indexes=List(list=orb_index_list),
            orbital_index_ref=Str('half_num_el'),
            edge_space=self.inputs.edge_space,
            dx=Float(0.15),
            retrieve_cubes=Bool(False),
            cubegen_parser_name=self.inputs.cubegen_parser_name,
            cubegen_parser_params=self.inputs.cubegen_parser_params)
        return ToContext(cubes=future)

    def finalize(self):

        if not common_utils.check_if_calc_ok(self, self.ctx.scf):
            return self.exit_codes.ERROR_TERMINATION

        self.report("Finalizing...")

        if self.should_do_cubes():
            if not common_utils.check_if_calc_ok(self, self.ctx.cubes):
                return self.exit_codes.ERROR_TERMINATION
            for cubes_out in list(self.ctx.cubes.outputs):
                if cubes_out.startswith("cube"):
                    self.out(cubes_out, self.ctx.cubes.outputs[cubes_out])
                elif cubes_out == 'retrieved':
                    self.out("cubes_retrieved",
                             self.ctx.cubes.outputs[cubes_out])

        self.out("energy_ev", self.ctx.scf.outputs.energy_ev)
        self.out("output_parameters",
                 self.ctx.scf.outputs['output_parameters'])

        self.out("remote_folder", self.ctx.scf.outputs['remote_folder'])

        return ExitCode(0)
