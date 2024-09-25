import ase
import numpy as np
from aiida import engine, orm

from ...utils import common_utils, cycle_tools, nmr
from .relax_workchain import GaussianRelaxWorkChain
from .scf_workchain import GaussianScfWorkChain


@engine.calcfunction
def prepare_nmr_structure(structure, height):
    ase_geo = structure.get_ase()
    pos = ase_geo.positions
    extents = np.array(
        [
            np.max(pos[:, 0]) - np.min(pos[:, 0]),
            np.max(pos[:, 1]) - np.min(pos[:, 1]),
            np.max(pos[:, 2]) - np.min(pos[:, 2]),
        ]
    )
    inds = np.argsort(-extents)
    new_pos = pos[:, inds]
    ase_atoms = ase.Atoms(numbers=ase_geo.numbers, positions=new_pos)
    ase_atoms_no_h = ase.Atoms([a for a in ase_atoms if a.symbol != "H"])
    cycles = cycle_tools.dumb_cycle_detection(ase_atoms_no_h, 8)
    ref_p = nmr.find_ref_points(ase_atoms_no_h, cycles, height.value)
    new_ase = ase_atoms + ref_p
    return orm.StructureData(ase=new_ase)


class GaussianNicsWorkChain(engine.WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("gaussian_code", valid_type=orm.Code)

        spec.input(
            "structure",
            valid_type=orm.StructureData,
            required=True,
            help="input geometry",
        )
        spec.input(
            "opt",
            valid_type=orm.Bool,
            required=False,
            help="False do not optimize structure",
        )
        spec.input(
            "height",
            valid_type=orm.Float,
            required=False,
            help="Height of NICS centers",
        )
        spec.input(
            "functional", valid_type=orm.Str, required=True, help="xc functional"
        )

        spec.input("basis_set", valid_type=orm.Str, required=True, help="basis_set")

        spec.input(
            "multiplicity",
            valid_type=orm.Int,
            required=False,
            default=lambda: orm.Int(0),
            help="spin multiplicity; 0 means RKS",
        )
        spec.input(
            "wfn_stable_opt",
            valid_type=orm.Bool,
            required=False,
            default=lambda: orm.Bool(False),
            help="if true, perform wfn stability optimization",
        )
        spec.input(
            "empirical_dispersion",
            valid_type=orm.Str,
            required=False,
            default=lambda: orm.Str(""),
            help=("Include empirical dispersion corrections" '(e.g. "GD3", "GD3BJ")'),
        )
        spec.input(
            "options",
            valid_type=orm.Dict,
            required=False,
            help="Use custom metadata.options instead of the automatic ones.",
        )

        spec.outline(
            cls.setup,
            engine.if_(cls.should_submit_opt)(cls.submit_opt, cls.inspect_opt),
            cls.submit_nics,
            cls.inspect_nics,
            cls.finalize,
        )

        spec.outputs.dynamic = True

        spec.exit_code(
            390,
            "ERROR_TERMINATION",
            message="One or more steps of the work chain failed.",
        )

    def setup(self):
        self.report("Setting up...")
        self.ctx.nmr_structure = self.inputs.structure
        self.ctx_should_opt = getattr(self.inputs, "opt", True)

    def should_submit_opt(self):
        return self.ctx_should_opt

    def submit_opt(self):
        self.report("Submitting optimization...")
        label = "geo_opt"
        builder = GaussianRelaxWorkChain.get_builder()
        builder.gaussian_code = self.inputs.gaussian_code
        builder.structure = self.inputs.structure
        builder.functional = self.inputs.functional
        builder.empirical_dispersion = self.inputs.empirical_dispersion
        builder.basis_set = self.inputs.basis_set
        builder.multiplicity = self.inputs.multiplicity
        builder.tight = orm.Bool(True)
        builder.int = orm.Str("superfine")
        builder.cdiis = orm.Bool(True)
        builder.maxcycle = orm.Int(2048)
        builder.conver = orm.Int(8)
        if "options" in self.inputs:
            builder.options = self.inputs.options

        submitted_node = self.submit(builder)
        submitted_node.description = label
        self.to_context(**{label: submitted_node})

    def inspect_opt(self):
        self.report("Inspecting optimization...")
        label = "geo_opt"
        # check if everything finished nicely
        if not common_utils.check_if_calc_ok(self, self.ctx[label]):
            return self.exit_codes.ERROR_TERMINATION
        self.ctx.nmr_structure = self.ctx[label].outputs.output_structure
        return engine.ExitCode(0)

    def submit_nics(self):
        self.report("Submitting NICS calculation...")
        label = "nics"
        builder = GaussianScfWorkChain.get_builder()
        height = getattr(self.inputs, "height", 1.0)
        builder.gaussian_code = self.inputs.gaussian_code
        builder.structure = prepare_nmr_structure(
            self.ctx.nmr_structure, orm.Float(height)
        )
        builder.functional = self.inputs.functional
        builder.basis_set = self.inputs.basis_set
        builder.multiplicity = self.inputs.multiplicity
        builder.int = orm.Str("superfine")
        builder.cdiis = orm.Bool(True)
        builder.nmr = orm.Bool(True)
        builder.maxcycle = orm.Int(2048)
        builder.conver = orm.Int(8)
        if "options" in self.inputs:
            builder.options = self.inputs.options

        submitted_node = self.submit(builder)
        submitted_node.description = label
        self.to_context(**{label: submitted_node})

    def inspect_nics(self):
        self.report("Inspecting nics...")
        label = "nics"
        # check if everything finished nicely
        if not common_utils.check_if_calc_ok(self, self.ctx[label]):
            return self.exit_codes.ERROR_TERMINATION

        self.out("output_parameters", self.ctx[label].outputs.output_parameters)
        self.out("output_structure", self.ctx.nics.inputs.structure)
        return engine.ExitCode(0)

    def finalize(self):
        self.report("Finalizing...")

        # Add extras.
        struc = self.inputs.structure
        common_utils.add_extras(struc, "surfaces", self.node.uuid)
