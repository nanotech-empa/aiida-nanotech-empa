from aiida import orm
from aiida.engine import WorkChain
from aiida.plugins import DataFactory, WorkflowFactory

StructureData = DataFactory('structure')
GaussianRelaxWorkChain = WorkflowFactory('nanotech_empa.gaussian.relax')


class GaussianVibronicStmWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("gaussian_code", valid_type=orm.Code)
        spec.input('structure',
                   valid_type=StructureData,
                   required=True,
                   help='input geometry')
        spec.input('functional',
                   valid_type=orm.Str,
                   required=True,
                   help='xc functional')
        spec.input('empirical_dispersion',
                   valid_type=orm.Str,
                   required=False,
                   default=lambda: orm.Str(""),
                   help=('Include empirical dispersion corrections'
                         '(e.g. "GD3", "GD3BJ")'))
        spec.input('basis_set',
                   valid_type=orm.Str,
                   required=True,
                   help='Basis set for the optimization.')
        spec.input(
            'multiplicity_initial',
            valid_type=orm.Int,
            required=False,
            default=lambda: orm.Int(1),
            help='Spin multiplicity of the neutral configuration. 0 means RKS.'
        )
        spec.input('charge_initial',
                   valid_type=orm.Int,
                   required=False,
                   default=lambda: orm.Int(0),
                   help='The charge of the charged configuration.')
        spec.input(
            'multiplicity_final',
            valid_type=orm.Int,
            required=False,
            default=lambda: orm.Int(2),
            help='Spin multiplicity of the charged configuration. 0 means RKS.'
        )
        spec.input('charge_final',
                   valid_type=orm.Int,
                   required=False,
                   default=lambda: orm.Int(-1),
                   help='The charge of the charged configuration.')
        spec.input(
            'options',
            valid_type=orm.Dict,
            required=False,
            help="Use custom metadata.options instead of the automatic ones.")
        spec.outline(cls.submit_opts, cls.finalize)
        spec.outputs.dynamic = True

    def submit_opts(self):

        # Neutral molecule
        builder = GaussianRelaxWorkChain.get_builder()
        builder.gaussian_code = self.inputs.gaussian_code
        builder.structure = self.inputs.structure
        builder.functional = self.inputs.functional
        builder.empirical_dispersion = self.inputs.empirical_dispersion
        builder.basis_set = self.inputs.basis_set
        builder.charge = self.inputs.charge_initial
        builder.multiplicity = self.inputs.multiplicity_initial
        if 'options' in self.inputs:
            builder.options = self.inputs.options
        submitted_node = self.submit(builder)
        self.to_context(**{'opt_neutral': submitted_node})

        # Charged molecule
        builder = GaussianRelaxWorkChain.get_builder()
        builder.gaussian_code = self.inputs.gaussian_code
        builder.structure = self.inputs.structure
        builder.functional = self.inputs.functional
        builder.empirical_dispersion = self.inputs.empirical_dispersion
        builder.basis_set = self.inputs.basis_set
        builder.freq = orm.Bool(True)
        builder.charge = self.inputs.charge_final
        builder.multiplicity = self.inputs.multiplicity_final
        if 'options' in self.inputs:
            builder.options = self.inputs.options
        submitted_node = self.submit(builder)
        self.to_context(**{'opt_charged': submitted_node})

    def finalize(self):
        self.out("neutral_out", self.ctx.opt_neutral.outputs.output_parameters)
        self.out("charged_out", self.ctx.opt_charged.outputs.output_parameters)
        self.report("Finalizing...")
