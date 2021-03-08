from aiida_nanotech_empa.workflows.gaussian import common

import numpy as np

from aiida.engine import WorkChain, ToContext, calcfunction, ExitCode
from aiida.orm import Code, Dict, RemoteData

from aiida.plugins import WorkflowFactory

GaussianBaseWorkChain = WorkflowFactory('gaussian.base')

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

        spec.outline(cls.submit_calc, cls.finalize)

        spec.outputs.dynamic = True

        spec.exit_code(
            390,
            "ERROR_TERMINATION",
            message="One or more steps of the work chain failed.",
        )

    def submit_calc(self):

        self.ctx.n_atoms = self.inputs.parent_calc_params['natom']
        basis_set = self.inputs.parent_calc_params['metadata']['basis_set']

        num_cores, memory_mb = common.determine_comp_resources(
            self.ctx.n_atoms, basis_set)

        builder = GaussianBaseWorkChain.get_builder()
        builder.gaussian.code = self.inputs.gaussian_code
        builder.gaussian.parent_calc_folder = self.inputs.parent_calc_folder
        builder.gaussian.parameters = Dict(
            dict={
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
            })

        common.set_metadata(builder.gaussian.metadata, self.ctx.n_atoms,
                            self.inputs.gaussian_code.computer)

        submitted_node = self.submit(builder)
        submitted_node.description = "naturalorbitals population"
        return ToContext(natorb=submitted_node)

    def finalize(self):

        if not common.check_if_previous_calc_ok(self, self.ctx.natorb):
            return self.exit_codes.ERROR_TERMINATION

        self.out("natorb_raw_parameters",
                 self.ctx.natorb.outputs.output_parameters)
        self.out(
            "natorb_proc_parameters",
            process_natural_orb_occupations(
                self.ctx.natorb.outputs.output_parameters))

        return ExitCode(0)
