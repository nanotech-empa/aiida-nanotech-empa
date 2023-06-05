import numpy as np
from aiida import engine, orm

from .geo_opt_workchain import Cp2kGeoOptWorkChain
from .molecule_gw_workchain import Cp2kMoleculeGwWorkChain


@engine.calcfunction
def analyze_structure(structure, mag_per_site):
    mol_atoms = structure.get_ase()

    mps = []
    if list(mag_per_site):
        mol_at_tuples = [
            (e, *np.round(p, 2))
            for e, p in zip(mol_atoms.get_chemical_symbols(), mol_atoms.positions)
        ]
        mps = [
            m
            for at, m in zip(mol_atoms, list(mag_per_site))
            if (at.symbol, *np.round(at.position, 2)) in mol_at_tuples
        ]

    return {
        "mol_struct": orm.StructureData(ase=mol_atoms),
        "mol_mag_per_site": orm.List(mps),
    }


class Cp2kMoleculeOptGwWorkChain(engine.WorkChain):
    """WorkChain to  optimize molecule and run GW:

    Two different ways to run:
    1) optimize geo and run gw
    2) run gw
    """

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("code", valid_type=orm.Code)

        spec.input(
            "structure", valid_type=orm.StructureData, help="An isolated molecule."
        )
        spec.input(
            "protocol",
            valid_type=orm.Str,
            default=lambda: orm.Str("gpw_std"),
            required=False,
            help="Protocol supported by the GW workchain.",
        )
        spec.input(
            "multiplicity",
            valid_type=orm.Int,
            default=lambda: orm.Int(0),
            required=False,
        )
        spec.input(
            "magnetization_per_site",
            valid_type=orm.List,
            default=lambda: orm.List(list=[]),
            required=False,
        )
        spec.input_namespace(
            "options",
            valid_type=dict,
            non_db=True,
            required=False,
            help="Define options for the cacluations: walltime, memory, CPUs, etc.",
        )

        spec.input(
            "options.geo_opt",
            valid_type=dict,
            non_db=True,
            required=False,
            help="Define options for the GEO_OPT cacluation: walltime, memory, CPUs, etc.",
        )

        spec.input(
            "options.gw",
            valid_type=dict,
            non_db=True,
            required=False,
            help="Define options for the GW cacluation: walltime, memory, CPUs, etc.",
        )
        spec.input(
            "debug",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            required=False,
            help="Run with fast parameters for debugging.",
        )
        spec.input(
            "geo_opt",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(True),
            required=False,
            help="Perform geo opt step.",
        )

        spec.outline(
            cls.setup,
            engine.if_(cls.gas_opt_selected)(cls.gas_opt, cls.check_gas_opt),
            cls.gw,
            cls.finalize,
        )
        spec.outputs.dynamic = True

        spec.exit_code(
            390,
            "ERROR_TERMINATION",
            message="One or more steps of the work chain failed.",
        )

    def setup(self):
        self.report("Inspecting input and setting up things.")

        n_atoms = len(self.inputs.structure.get_ase())
        n_mags = len(list(self.inputs.magnetization_per_site))
        if n_mags not in (0, n_atoms):
            self.report("If set, magnetization_per_site needs a value for every atom.")
            return self.exit_codes.ERROR_TERMINATION

        an_out = analyze_structure(
            self.inputs.structure, self.inputs.magnetization_per_site
        )

        self.ctx.mol_struct = an_out["mol_struct"]
        self.ctx.mol_mag_per_site = an_out["mol_mag_per_site"]

        return engine.ExitCode(0)

    def gas_opt_selected(self):
        return self.inputs.geo_opt.value

    def gas_opt(self):
        builder = Cp2kGeoOptWorkChain.get_builder()
        builder.code = self.inputs.code
        builder.structure = self.ctx.mol_struct
        builder.multiplicity = self.inputs.multiplicity
        builder.magnetization_per_site = self.ctx.mol_mag_per_site
        builder.vdw = orm.Bool(True)
        builder.protocol = orm.Str("standard")
        builder.options = self.inputs.options.geo_opt
        builder.metadata.description = "Submitted by Cp2kMoleculeOptGwWorkChain."
        builder.metadata.label = "Cp2kGeoOptWorkChain"
        return engine.ToContext(gas_opt=self.submit(builder))

    def check_gas_opt(self):
        if not self.ctx.gas_opt.is_finished_ok:
            return self.exit_codes.ERROR_TERMINATION

        # Set the optimized geometry as ctx geometry.
        self.ctx.mol_struct = self.ctx.gas_opt.outputs.output_structure
        return engine.ExitCode(0)

    def gw(self):
        self.report("Submitting GW.")

        builder = Cp2kMoleculeGwWorkChain.get_builder()
        builder.code = self.inputs.code
        builder.protocol = self.inputs.protocol
        builder.structure = self.ctx.mol_struct
        builder.magnetization_per_site = self.ctx.mol_mag_per_site
        builder.multiplicity = self.inputs.multiplicity
        builder.debug = self.inputs.debug
        builder.options.scf = self.inputs.options.geo_opt
        builder.options.gw = self.inputs.options.gw
        builder.metadata.description = "gw"
        submitted_node = self.submit(builder)
        return engine.ToContext(gw=submitted_node)

    def finalize(self):
        self.report("Finalizing...")

        if not self.ctx.gw.is_finished_ok:
            return self.exit_codes.ERROR_TERMINATION

        gw_out_params = self.ctx.gw.outputs.gw_output_parameters
        self.out("gw_output_parameters", gw_out_params)

        self.out("output_structure", self.ctx.mol_struct)

        # Add the workchain pk to the input/geo_opt structure extras.
        struc_to_label = self.ctx.mol_struct
        extras_label = "Cp2kMoleculeOptGwWorkChain_pks"
        if extras_label not in struc_to_label.base.extras.all:
            extras_list = []
        else:
            extras_list = struc_to_label.base.extras.all[extras_label]
        extras_list.append(self.node.pk)
        struc_to_label.base.extras.set(extras_label, extras_list)

        return engine.ExitCode(0)
