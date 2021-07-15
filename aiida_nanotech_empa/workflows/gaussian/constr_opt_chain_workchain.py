from aiida_nanotech_empa.utils import common_utils

from aiida.engine import WorkChain, ToContext, ExitCode, while_
from aiida.orm import Int, Str, Code, Bool, Dict, List, StructureData

from aiida.plugins import WorkflowFactory

GaussianRelaxWorkChain = WorkflowFactory('nanotech_empa.gaussian.relax')
GaussianScfWorkChain = WorkflowFactory('nanotech_empa.gaussian.scf')


class GaussianConstrOptChainWorkChain(WorkChain):
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

        spec.input('basis_set_scf',
                   valid_type=Str,
                   required=False,
                   help='basis_set for SCF')

        spec.input('multiplicity',
                   valid_type=Int,
                   required=False,
                   default=lambda: Int(0),
                   help='spin multiplicity; 0 means RKS')

        spec.input('extra_scf_mults',
                   valid_type=List,
                   required=False,
                   default=lambda: List(list=[]),
                   help='Extra multiplicites for the SCF')

        spec.input('tight',
                   valid_type=Bool,
                   required=False,
                   default=lambda: Bool(False),
                   help='Use tight optimization criteria.')

        spec.input('empirical_dispersion',
                   valid_type=Str,
                   required=False,
                   default=lambda: Str(""),
                   help=('Include empirical dispersion corrections'
                         '(e.g. "GD3", "GD3BJ")'))

        spec.input('list_of_constraints',
                   valid_type=List,
                   required=False,
                   default=lambda: List(list=[]),
                   help='Supported constraints: ("distance", n1, n2, d)')

        spec.input(
            'options',
            valid_type=Dict,
            required=False,
            help="Use custom metadata.options instead of the automatic ones.")

        spec.outline(
            cls.setup,
            while_(cls.any_constraint_left)(cls.submit_opt,
                                            cls.submit_extra_mults),
            cls.finalize)

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

    def setup(self):
        self.ctx.i_constr = 0

    def any_constraint_left(self):
        return len(self.inputs.list_of_constraints) > self.ctx.i_constr

    def submit_opt(self):

        label = f"opt_{self.ctx.i_constr}"
        if self.ctx.i_constr == 0:
            structure = self.inputs.structure
        else:
            prev_label = f"opt_{self.ctx.i_constr-1}"

            for extra_mult in self.inputs.extra_scf_mults:
                ext_label = prev_label + f"_m{extra_mult}"
                if not common_utils.check_if_calc_ok(self,
                                                     self.ctx[ext_label]):
                    return self.exit_codes.ERROR_TERMINATION

            structure = self.ctx[prev_label].outputs.output_structure

        cur_constr = self.inputs.list_of_constraints[self.ctx.i_constr]

        self.ctx.i_constr += 1

        self.report(f"Submitting optimization {label}")

        builder = GaussianRelaxWorkChain.get_builder()
        builder.gaussian_code = self.inputs.gaussian_code
        builder.structure = structure
        builder.functional = self.inputs.functional
        builder.basis_set = self.inputs.basis_set

        if 'basis_set_scf' in self.inputs:
            builder.basis_set_scf = self.inputs.basis_set_scf

        builder.multiplicity = self.inputs.multiplicity
        builder.wfn_stable_opt = Bool(self.is_uks())

        builder.empirical_dispersion = self.inputs.empirical_dispersion

        builder.tight = self.inputs.tight

        builder.constraints = List(list=cur_constr)

        if 'options' in self.inputs:
            builder.options = self.inputs.options

        submitted_node = self.submit(builder)
        submitted_node.description = label
        return ToContext(**{label: submitted_node})

    def submit_extra_mults(self):

        opt_label = f"opt_{self.ctx.i_constr-1}"

        if not common_utils.check_if_calc_ok(self, self.ctx[opt_label]):
            return self.exit_codes.ERROR_TERMINATION

        for extra_mult in self.inputs.extra_scf_mults:

            label = opt_label + f"_m{extra_mult}"
            self.report(f"Submitting scf {label}")

            builder = GaussianScfWorkChain.get_builder()
            builder.gaussian_code = self.inputs.gaussian_code
            builder.structure = self.ctx[opt_label].outputs.output_structure
            builder.functional = self.inputs.functional
            builder.empirical_dispersion = self.inputs.empirical_dispersion
            builder.basis_set = self.inputs.basis_set_scf
            builder.multiplicity = Int(extra_mult)
            builder.wfn_stable_opt = Bool(True)

            if 'options' in self.inputs:
                builder.options = self.inputs.options

            submitted_node = self.submit(builder)
            submitted_node.description = label
            self.to_context(**{label: submitted_node})

        return ExitCode(0)

    def finalize(self):

        self.report("Finalizing...")

        for i_constr in range(len(self.inputs.list_of_constraints)):

            label = f"opt_{i_constr}"
            if not common_utils.check_if_calc_ok(self, self.ctx[label]):
                return self.exit_codes.ERROR_TERMINATION

            self.out(f"opt_{i_constr}_structure",
                     self.ctx[label].outputs.output_structure)
            self.out(f"opt_{i_constr}_out_params",
                     self.ctx[label].outputs.output_parameters)
            if 'scf_output_parameters' in self.ctx[label].outputs:
                self.out(f"opt_{i_constr}_scf_out_params",
                         self.ctx[label].outputs.scf_output_parameters)

            for extra_mult in self.inputs.extra_scf_mults:
                extra_label = label + f"_m{extra_mult}"
                self.out(f"{extra_label}_scf_out_params",
                         self.ctx[extra_label].outputs.output_parameters)

        return ExitCode(0)
