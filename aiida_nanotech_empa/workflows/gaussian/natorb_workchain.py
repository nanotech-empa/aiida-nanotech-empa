from aiida_nanotech_empa.workflows.gaussian import common
from aiida_nanotech_empa.utils import common_utils

import numpy as np

from aiida.engine import WorkChain, ToContext, calcfunction, ExitCode, if_
from aiida.orm import Code, Dict, RemoteData, Bool, Int, List, Float

from aiida.plugins import WorkflowFactory

GaussianBaseWorkChain = WorkflowFactory('gaussian.base')
GaussianCubesWorkChain = WorkflowFactory('gaussian.cubes')

## --------------------------------------------------------------------
## Natural orbital processing:


def standard_num_odd(no_occs):
    n_odd = 0.0
    for n in no_occs:
        n_odd += 2 * n - n**2
    return n_odd


def head_gordon_num_odd(no_occs):
    n_odd = 0.0
    for n in no_occs:
        n_odd += 1 - np.abs(1 - n)
    return n_odd


def head_gordon_alt_num_odd(no_occs):
    n_odd = 0.0
    for n in no_occs:
        n_odd += n**2 * (2 - n)**2
    return n_odd


def spin_proj_nakano(no_occs, i_hono=None):
    # "perfect-pairing spin projection scheme"
    #     Nakano 2011: (Hyper)polarizability density analysis...
    # More recent citation, also reveals the connection to Yamaguchi's scheme:
    #     Nakano 2015: Approximate spin projected spin-unrestricted...

    # Equivalent to "Yamaguchi's scheme"
    # Original citation:
    #    Yamaguchi 1988: A spin correction procedure...
    #    (No radical character BUT singlet-triplet energy gap correction)
    # Recent citation:
    #    Minami, Nakano 2012: Diradical Character View of Singlet Fission
    #    (Radical and multiradical characters)
    # An application paper:
    #    Lu 2016: Stable 3,6-Linked Fluorenyl Radical Oligomers with...

    if i_hono is None:
        no_hono = no_occs[no_occs > 1.0]
        no_luno = no_occs[no_occs <= 1.0]
    else:
        no_hono = no_occs[:i_hono + 1]
        no_luno = no_occs[i_hono + 1:]

    c = np.min([len(no_hono), len(no_luno)])

    no_hono = no_hono[::-1]
    # overlap between pairs
    s = (no_hono[:c] - no_luno[:c]) / 2

    no_hono_sp = no_hono[:c]**2 / (1 + s**2)
    no_luno_sp = no_luno[:c]**2 / (1 + s**2)

    # pad the spin proj array to initial array shape
    no_hono_sp = np.pad(no_hono_sp, (0, len(no_hono) - len(no_hono_sp)),
                        mode='constant',
                        constant_values=2.0)
    no_luno_sp = np.pad(no_luno_sp, (0, len(no_luno) - len(no_luno_sp)),
                        mode='constant',
                        constant_values=0.0)

    return np.concatenate([no_hono_sp[::-1], no_luno_sp])


@calcfunction
def process_natural_orb_occupations(natorb_parameters):

    no_occs = natorb_parameters['nooccnos']
    i_homo = natorb_parameters['homos'][0]
    no_occs_sp = list(spin_proj_nakano(np.array(no_occs), i_hono=i_homo))

    return Dict(
        dict={
            'no_occs': no_occs,
            'no_occs_sp': no_occs_sp,
            'std_num_odd': standard_num_odd(no_occs),
            'std_num_odd_sp': standard_num_odd(no_occs_sp),
            'hg_num_odd': head_gordon_num_odd(no_occs),
            'hg_num_odd_sp': head_gordon_num_odd(no_occs_sp),
        })


## --------------------------------------------------------------------


class GaussianNatOrbWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("gaussian_code", valid_type=Code)

        spec.input(
            "parent_calc_folder",
            valid_type=RemoteData,
            required=True,
            help="parent Gaussian calculation directory",
        )

        spec.input(
            "parent_calc_params",
            valid_type=Dict,
            required=True,
            help="parent Gaussian calculation output parameters",
        )

        spec.input("save_natorb_chk",
                   valid_type=Bool,
                   required=False,
                   default=lambda: Bool(False),
                   help=("Save natural orbitals in the chk file." +
                         "Can introduce errors for larger systems"))

        # -------------------------------------------------------------------
        # CUBE GENERATION INPUTS
        spec.input(
            "num_natural_orbital_cubes",
            valid_type=Int,
            required=False,
            default=lambda: Int(0),
            help='Generate cubes for SAVED natural orbitals (n*occ and n*virt).'
        )
        spec.input("formchk_code", valid_type=Code, required=False)
        spec.input("cubegen_code", valid_type=Code, required=False)

        spec.input(
            'edge_space',
            valid_type=Float,
            required=False,
            default=lambda: Float(3.0),
            help='Extra cube space in addition to molecule bounding box [ang].'
        )
        spec.input("cubegen_parser_params",
                   valid_type=Dict,
                   required=False,
                   default=lambda: Dict(dict={}),
                   help='Additional parameters to cubegen parser.')
        # -------------------------------------------------------------------

        spec.input(
            "options",
            valid_type=Dict,
            required=False,
            help="Use custom metadata.options instead of the automatic ones.",
        )

        spec.outline(
            cls.submit_calc,
            if_(cls.save_natorb_chk)(cls.submit_save,
                                     if_(cls.should_do_cubes)(cls.cubes)),
            cls.finalize)

        spec.outputs.dynamic = True

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

    def submit_calc(self):

        self.ctx.n_atoms = self.inputs.parent_calc_params['natom']
        self.ctx.basis_set = self.inputs.parent_calc_params['metadata'][
            'basis_set']
        self.ctx.comp = self.inputs.gaussian_code.computer

        success = common.determine_metadata_options(self)
        if not success:
            return self.exit_codes.ERROR_OPTIONS

        num_cores, memory_mb = common.get_gaussian_cores_and_memory(
            self.ctx.metadata_options, self.ctx.comp)

        self.ctx.num_cores = num_cores
        self.ctx.memory_mb = memory_mb

        builder = GaussianBaseWorkChain.get_builder()
        builder.gaussian.code = self.inputs.gaussian_code
        builder.gaussian.parent_calc_folder = self.inputs.parent_calc_folder

        parameters = {
            'link0_parameters': {
                '%chk': 'aiida.chk',
                '%oldchk': 'parent_calc/aiida.chk',
                '%mem': "%dMB" % memory_mb,
                '%nprocshared': str(num_cores),
            },
            'route_parameters': {
                'guess': {
                    'read': None,
                    'only': None,
                },
                'pop': 'naturalorbital',
                'geom': 'allcheck',
                'chkbasis': None,
            },
            'functional': "",  # ignored
            'basis_set': "",  # ignored
            'charge': -1,  # ignored
            'multiplicity': -1,  # ignored
        }

        builder.gaussian.parameters = Dict(dict=parameters)

        builder.gaussian.metadata.options = self.ctx.metadata_options

        submitted_node = self.submit(builder)
        submitted_node.description = "naturalorbitals population"
        return ToContext(natorb=submitted_node)

    def save_natorb_chk(self):
        return self.inputs.save_natorb_chk

    def submit_save(self):

        if not common_utils.check_if_calc_ok(self, self.ctx.natorb):
            return self.exit_codes.ERROR_TERMINATION

        builder = GaussianBaseWorkChain.get_builder()
        builder.gaussian.code = self.inputs.gaussian_code
        #builder.gaussian.parent_calc_folder = self.ctx.natorb.outputs.remote_folder
        builder.gaussian.parent_calc_folder = self.inputs.parent_calc_folder

        parameters = {
            'link0_parameters': {
                '%chk': 'aiida.chk',
                '%oldchk': 'parent_calc/aiida.chk',
                '%mem': "%dMB" % self.ctx.memory_mb,
                '%nprocshared': str(self.ctx.num_cores),
            },
            'route_parameters': {
                'guess': {
                    'save': None,
                    'only': None,
                    'naturalorbitals': None,
                },
                'geom': 'allcheck',
                'chkbasis': None,
            },
            'functional': "",  # ignored
            'basis_set': "",  # ignored
            'charge': -1,  # ignored
            'multiplicity': -1,  # ignored
        }

        builder.gaussian.parameters = Dict(dict=parameters)

        builder.gaussian.metadata.options = self.ctx.metadata_options

        submitted_node = self.submit(builder)
        submitted_node.description = "naturalorbitals save"
        return ToContext(natorb_save=submitted_node)

    def should_do_cubes(self):
        codes_set = 'formchk_code' in self.inputs and 'cubegen_code' in self.inputs
        pos_num_specified = self.inputs.num_natural_orbital_cubes.value > 0
        return self.save_natorb_chk() and codes_set and pos_num_specified

    def cubes(self):

        if not common_utils.check_if_calc_ok(self, self.ctx.natorb_save):
            return self.exit_codes.ERROR_TERMINATION

        n_d = self.inputs.num_natural_orbital_cubes.value
        n_u = self.inputs.num_natural_orbital_cubes.value

        builder = GaussianCubesWorkChain.get_builder()
        builder.formchk_code = self.inputs.formchk_code
        builder.cubegen_code = self.inputs.cubegen_code
        builder.gaussian_calc_folder = self.ctx.natorb_save.outputs.remote_folder
        builder.gaussian_output_params = self.ctx.natorb.outputs.output_parameters
        builder.orbital_indexes = List(list=list(range(-n_d + 1, n_u + 1)))
        builder.natural_orbitals = Bool(True)
        builder.edge_space = self.inputs.edge_space
        builder.dx = Float(0.15)
        builder.cubegen_parser_name = 'nanotech_empa.gaussian.cubegen_pymol'
        builder.cubegen_parser_params = self.inputs.cubegen_parser_params

        future = self.submit(builder)
        return ToContext(cubes=future)

    def finalize(self):

        if not common_utils.check_if_calc_ok(self, self.ctx.natorb):
            return self.exit_codes.ERROR_TERMINATION

        if self.save_natorb_chk():
            if not common_utils.check_if_calc_ok(self, self.ctx.natorb_save):
                return self.exit_codes.ERROR_TERMINATION
            self.out('remote_folder',
                     self.ctx.natorb_save.outputs.remote_folder)
        else:
            self.out('remote_folder', self.ctx.natorb.outputs.remote_folder)

        if self.should_do_cubes():
            if not common_utils.check_if_calc_ok(self, self.ctx.cubes):
                return self.exit_codes.ERROR_TERMINATION
            for cubes_out in list(self.ctx.cubes.outputs):
                if cubes_out.startswith("cube"):
                    self.out(cubes_out, self.ctx.cubes.outputs[cubes_out])

        self.out("natorb_raw_parameters",
                 self.ctx.natorb.outputs.output_parameters)
        self.out(
            "natorb_proc_parameters",
            process_natural_orb_occupations(
                self.ctx.natorb.outputs.output_parameters))

        return ExitCode(0)
