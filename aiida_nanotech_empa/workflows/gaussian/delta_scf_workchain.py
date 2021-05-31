from aiida_nanotech_empa.workflows.gaussian import common
from aiida_nanotech_empa.utils import common_utils

from aiida.engine import WorkChain, calcfunction, ExitCode
from aiida.orm import Int, Str, Code, Dict
from aiida.orm import StructureData, RemoteData

from aiida.plugins import WorkflowFactory

GaussianBaseWorkChain = WorkflowFactory('gaussian.base')


@calcfunction
def subtract(a, b):
    return a - b


class GaussianDeltaScfWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("gaussian_code", valid_type=Code)

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

        spec.input('parent_calc_folder',
                   valid_type=RemoteData,
                   required=False,
                   help="the folder of a completed gaussian calculation")

        spec.input(
            'options',
            valid_type=Dict,
            required=False,
            help="Use custom metadata.options instead of the automatic ones.")

        spec.outline(cls.setup, cls.submit_scfs, cls.finalize)

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
        self.report("Inspecting input and setting up things.")

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

    def setup_common_builder_params(self, builder):
        builder.gaussian.structure = self.inputs.structure
        builder.gaussian.code = self.inputs.gaussian_code
        builder.gaussian.metadata.options = self.ctx.metadata_options

    def submit_scfs(self):
        # pylint: disable=too-many-branches

        # --------------------------------------------------
        self.report("Submitting NEUTRAL SCF")
        # --------------------------------------------------

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
                    'sp': None,
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
        self.setup_common_builder_params(builder)
        submitted_node = self.submit(builder)
        self.to_context(neutral=submitted_node)

        # --------------------------------------------------
        self.report("Submitting CATION SCF")
        # --------------------------------------------------

        if self.ctx.mult == 1:
            pos_mult = 2
        else:
            pos_mult = self.ctx.mult - 1

        if self.is_uks():
            functional = self.ctx.functional
        else:
            functional = 'u' + self.ctx.functional

        parameters = Dict(
            dict={
                'link0_parameters': self.ctx.link0.copy(),
                'functional': functional,
                'basis_set': self.inputs.basis_set.value,
                'charge': 1,
                'multiplicity': pos_mult,
                'route_parameters': {
                    'scf': {
                        'maxcycle': 128
                    },
                    'sp': None,
                },
            })

        builder = GaussianBaseWorkChain.get_builder()

        if pos_mult == 1:
            # For open-shell singlet, mix homo & lumo
            parameters['route_parameters']['guess'] = "mix"

        builder.gaussian.parameters = parameters
        self.setup_common_builder_params(builder)
        submitted_node = self.submit(builder)
        self.to_context(pos=submitted_node)

        # --------------------------------------------------
        self.report("Submitting ANION SCF")
        # --------------------------------------------------
        # For the ANION, the added electron could go opposite or parallel
        # if the system was already spin-polarized

        if self.ctx.mult == 1:
            self.ctx.neg_mults = [2]
        else:
            self.ctx.neg_mults = [self.ctx.mult - 1, self.ctx.mult + 1]

        for neg_mult in self.ctx.neg_mults:
            parameters = Dict(
                dict={
                    'link0_parameters': self.ctx.link0.copy(),
                    'functional': functional,
                    'basis_set': self.inputs.basis_set.value,
                    'charge': -1,
                    'multiplicity': neg_mult,
                    'route_parameters': {
                        'scf': {
                            'maxcycle': 128
                        },
                        'sp': None,
                    },
                })

            builder = GaussianBaseWorkChain.get_builder()

            if neg_mult == 1:
                # For open-shell singlet, mix homo & lumo
                parameters['route_parameters']['guess'] = "mix"

            builder.gaussian.parameters = parameters
            self.setup_common_builder_params(builder)
            submitted_node = self.submit(builder)
            label = f"neg_m{neg_mult}"
            self.to_context(**{label: submitted_node})

    def finalize(self):

        if (not common_utils.check_if_calc_ok(self, self.ctx.neutral)
                or not common_utils.check_if_calc_ok(self, self.ctx.pos)):
            return self.exit_codes.ERROR_TERMINATION

        anion_energies = []

        for neg_mult in self.ctx.neg_mults:
            label = f"neg_m{neg_mult}"
            if not common_utils.check_if_calc_ok(self, self.ctx[label]):
                return self.exit_codes.ERROR_TERMINATION

            anion_energies.append(self.ctx[label].outputs.energy_ev)

            if len(self.ctx.neg_mults) > 1:
                self.out(f"anion_energy_m{neg_mult}",
                         self.ctx[label].outputs.energy_ev)

        anion_energy = min(anion_energies)

        self.out("neutral_energy", self.ctx.neutral.outputs.energy_ev)
        self.out("cation_energy", self.ctx.pos.outputs.energy_ev)
        self.out("anion_energy", anion_energy)

        ip = subtract(self.ctx.pos.outputs.energy_ev,
                      self.ctx.neutral.outputs.energy_ev)
        ea = subtract(self.ctx.neutral.outputs.energy_ev, anion_energy)

        self.out("ionization_potential", ip)
        self.out("electron_affinity", ea)

        self.out("fundamental_gap", subtract(ip, ea))

        return ExitCode(0)
