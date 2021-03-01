from aiida_nanotech_empa.workflows.gaussian import common

from aiida.engine import WorkChain, ToContext, if_, ExitCode
from aiida.orm import Int, Str, Code, Dict, Bool, StructureData

from aiida.plugins import WorkflowFactory

GaussianBaseWorkChain = WorkflowFactory('gaussian.base')


class GaussianRelaxWorkChain(WorkChain):
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
                   default=lambda: Int(0),
                   help='spin multiplicity; 0 means RKS')

        spec.input('wfn_stable_opt',
                   valid_type=Bool,
                   required=False,
                   default=lambda: Bool(False),
                   help='if true, perform wfn stability optimization')

        spec.outline(cls.setup,
                     if_(cls.should_do_wfn_stability)(cls.uks_wfn_stability),
                     cls.optimization, cls.finalize)

        spec.outputs.dynamic = True

        spec.exit_code(
            301,
            "ERROR_MULTIPLICITY",
            message="Multiplicity and number of el. doesn't match.",
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

    def should_do_wfn_stability(self):
        if self.is_uks() and self.inputs.wfn_stable_opt:
            return True
        return False

    def setup(self):
        self.report("Inspecting input and setting up things")

        pymatgen_structure = self.inputs.structure.get_pymatgen_molecule()
        self.ctx.n_atoms = pymatgen_structure.num_sites
        n_electrons = pymatgen_structure.nelectrons

        if self.is_uks():
            self.ctx.functional = 'u' + self.inputs.functional.value
            self.ctx.mult = self.inputs.multiplicity.value
        else:
            self.ctx.functional = self.inputs.functional.value
            self.ctx.mult = 1

        if self.ctx.mult % 2 == n_electrons % 2:
            return self.exit_codes.ERROR_MULTIPLICITY

        num_cores, memory_mb = common.determine_comp_resources(
            self.ctx.n_atoms)

        self.ctx.num_cores = num_cores
        self.ctx.memory_mb = memory_mb

        self.ctx.link0 = {
            '%chk': 'aiida.chk',
            '%mem': "%dMB" % memory_mb,
            '%nprocshared': str(num_cores),
        }

        self.ctx.comp = self.inputs.gaussian_code.computer

        return ExitCode(0)

    def uks_wfn_stability(self):

        self.report("Running UKS WFN Stability")

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
                    'Stable': 'opt',
                },
            })

        if self.ctx.mult == 1:
            parameters['route_parameters']['guess'] = "mix"

        builder = GaussianBaseWorkChain.get_builder()

        builder.gaussian.parameters = parameters
        builder.gaussian.structure = self.inputs.structure
        builder.gaussian.code = self.inputs.gaussian_code

        common.set_metadata(builder.gaussian.metadata, self.ctx.n_atoms,
                            self.ctx.comp)

        future = self.submit(builder)
        return ToContext(uks_stab=future)

    def optimization(self):

        self.report("Submitting optimization")

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
                    'opt': 'tight',
                },
            })

        builder = GaussianBaseWorkChain.get_builder()

        if self.should_do_wfn_stability():
            parameters['link0_parameters']['%oldchk'] = "parent_calc/aiida.chk"
            parameters['route_parameters']['guess'] = "read"
            builder.gaussian.parent_calc_folder = self.ctx.uks_stab.outputs.remote_folder
        elif self.inputs.multiplicity == 1:
            parameters['route_parameters']['guess'] = "mix"

        builder.gaussian.parameters = parameters
        builder.gaussian.structure = self.inputs.structure
        builder.gaussian.code = self.inputs.gaussian_code

        common.set_metadata(builder.gaussian.metadata, self.ctx.n_atoms,
                            self.ctx.comp)

        future = self.submit(builder)
        return ToContext(opt=future)

    def finalize(self):

        if not common.check_if_previous_calc_ok(self, self.ctx.opt):
            return self.exit_codes.ERROR_TERMINATION

        self.report("Finalizing...")

        self.out("output_structure", self.ctx.opt.outputs.output_structure)
        self.out("energy_ev", self.ctx.opt.outputs.energy_ev)
        self.out("output_parameters", self.ctx.opt.outputs.output_parameters)
        self.out("remote_folder", self.ctx.opt.outputs.remote_folder)

        return ExitCode(0)
