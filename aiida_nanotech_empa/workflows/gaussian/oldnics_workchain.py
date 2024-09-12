import numpy as np
from aiida import engine, orm, plugins

from ...utils import common_utils, cycle_tools, nmr

#from .delta_scf_workchain import GaussianDeltaScfWorkChain
#from .natorb_workchain import GaussianNatOrbWorkChain
#from .relax_workchain import GaussianRelaxWorkChain
f#rom .scf_workchain import GaussianScfWorkChain
#from aiida.plugins import CalculationFactory

#GaussianBaseWorkChain = plugins.WorkflowFactory("gaussian.base")
#GaussianCubesWorkChain = plugins.WorkflowFactory("gaussian.cubes")
GaussianCalculation = plugins.CalculationFactory("gaussian")

@engine.calcfunction
def nics_structure(structure=None, h=1.0):

    ase_geo = structure.get_ase()
    # orient the geometry
    pos = ase_geo.positions
    extents = np.array([
        np.max(pos[:,0]) - np.min(pos[:,0]),
        np.max(pos[:,1]) - np.min(pos[:,1]),
        np.max(pos[:,2]) - np.min(pos[:,2]),
    ])
    inds = np.argsort(-extents)
    ase_atoms = ase.Atoms(numbers=ase_geo.numbers, positions=pos[:, inds])
    ase_atoms_no_h = ase.Atoms([a for a in ase_atoms if a.symbol != 'H'])
    cycles = cycle_tools.dumb_cycle_detection(ase_atoms_no_h, 8)
    ase_geom = ase_atoms + nmr.find_ref_points(ase_atoms_no_h, cycles, h.value)
    return orm.StructureData(ase=ase_geom)

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
            "functional", valid_type=orm.Str, required=False,default=lambda: orm.Str("b3lyp"), help="xc functional"
        )
        spec.input(
            "basis_set", valid_type=orm.Str, required=False, default=lambda: orm.Str("6-311+G(d,p)"), help="basis_set for opt"
        )
        spec.input(
            "multiplicity",
            valid_type=orm.Int,
            required=True,
            help="spin multiplicity",
        )

        spec.outline(
            cls.setup
            cls.submit_nics,
            cls.finalize,
        )

        spec.outputs.dynamic = True

        spec.exit_code(
            390,
            "ERROR_TERMINATION",
            message="One or more steps of the work chain failed.",
        )

    def setup(self):
        self.ctx.structure_wtih_centers = nics_structure(self.inputs.structure, h=1.0)
        self.ctx.parameters = {
                "link0_parameters": {
                    "%chk": "aiida.chk",
                    "%mem": "4096MB",
                    "%nprocshared": 4,
                },
                "functional": "BLYP",
                "basis_set": "6-311+G(d,p)",
                "multiplicity": self.inputs.multiplicity.value,
                "dieze_tag": "#P",
                "route_parameters": {
                    "scf": {
                        "maxcycle":2048,
                        "conver":8,
                    },
                    "int": "superfine",
                    "chpf":{"conver":8}
                    "nmr": None,
                },
            }

    def submit_nics(self):
        # Multiplicity 0 means RKS calculation.
        label = f"m{self.inputs.multiplicity.value}_opt"

        # Main parameters: geometry optimization
        parameters = Dict(
            {
                "link0_parameters": {
                    "%chk": "aiida.chk",
                    "%mem": "4096MB",
                    "%nprocshared": 4,
                },
                "functional": "BLYP",
                "basis_set": "6-311+G(d,p)",
                "multiplicity": self.inputs.multiplicity.value,
                "dieze_tag": "#P",
                "route_parameters": {
                    "scf": {
                        "maxcycle":2048,
                        "conver":8,
                    },
                    "int": "superfine",
                    "chpf":{"conver":8}
                    "nmr": None,
                },
            }
        )

        # Construct process builder

        builder = GaussianCalculation.get_builder()

        builder.structure = structure
        builder.parameters = parameters
        builder.code = gaussian_code

        builder.metadata.options.resources = {
            "num_machines": 1,
            "tot_num_mpiprocs": num_cores,
        }

        # Should ask for extra +25% extra memory
        builder.metadata.options.max_memory_kb = int(1.25 * memory_mb) * 1024
        builder.metadata.options.max_wallclock_seconds = 5 * 60

        submitted_node = self.submit(builder)
        submitted_node.description = label
        self.to_context(**{label: submitted_node})

    def inspect_opt(self):

        label = f"m{self.inputs.multiplicity.value}_opt"

        # check if everything finished nicely
        if not common_utils.check_if_calc_ok(self, self.ctx[label]):
            return self.exit_codes.ERROR_TERMINATION

        opt_energy = self.ctx[label].outputs.scf_energy_ev
        self.out("opt_energy", opt_energy)
        self.out("opt_structure", self.ctx[label].outputs.output_structure)
        self.out(
            "opt_out_params", self.ctx[label].outputs.scf_output_parameters
        )

        return engine.ExitCode(0)

    def submit_next_steps(self):

        self.report("Submitting NMR cubes")

        parameters = orm.Dict(
                    {
                        "link0_parameters": self.ctx.link0.copy(),
                        "dieze_tag": "#P",
                        "functional": self.ctx.functional,
                        "basis_set": self.inputs.basis_set.value,
                        "charge": 0,
                        "multiplicity": self.ctx.mult,
                        "route_parameters": {
                            "scf": {"maxcycle": 140},
                            "opt": None,
                        },
                    }
                )

        for mult in self.inputs.multiplicity_list:
            label = "nmr"

            if mult == self.ctx.gs_mult:
                continue

            builder = GaussianScfWorkChain.get_builder()
            builder.gaussian_code = self.inputs.gaussian_code
            builder.structure = self.ctx.gs_structure
            builder.functional = self.inputs.functional
            builder.empirical_dispersion = self.inputs.empirical_dispersion
            builder.basis_set =
            builder.multiplicity = self.inputs.multiplicity

            #if "options" in self.inputs:
            #    builder.options = self.inputs.options

            submitted_node = self.submit(builder)
            submitted_node.description = label
            self.to_context(**{label: submitted_node})

    def inspect_next_steps(self):
        if not common_utils.check_if_calc_ok(self, self.ctx.gs_cubes):
            return self.exit_codes.ERROR_TERMINATION

        self.out("gs_cube_images", self.ctx.gs_cubes.outputs.cube_image_folder)
        self.out("gs_cube_planes", self.ctx.gs_cubes.outputs.cube_planes_array)

        if not common_utils.check_if_calc_ok(self, self.ctx.dscf):
            return self.exit_codes.ERROR_TERMINATION

        self.out("gs_ionization_potential", self.ctx.dscf.outputs.ionization_potential)
        self.out("gs_electron_affinity", self.ctx.dscf.outputs.electron_affinity)

        for mult in self.inputs.multiplicity_list:
            label = f"m{mult}_vert"

            if mult == self.ctx.gs_mult:
                continue

            # Check if everything finished nicely.
            if not common_utils.check_if_calc_ok(self, self.ctx[label]):
                return self.exit_codes.ERROR_TERMINATION

            vert_energy = self.ctx[label].outputs.energy_ev
            self.out(f"m{mult}_vert_energy", vert_energy)
            self.out(
                f"m{mult}_vert_out_params", self.ctx[label].outputs.output_parameters
            )
            self.out(
                f"m{mult}_vert_cube_images", self.ctx[label].outputs.cube_image_folder
            )
            self.out(
                f"m{mult}_vert_cube_planes", self.ctx[label].outputs.cube_planes_array
            )

        return engine.ExitCode(0)

    def is_gs_oss(self):
        """Is ground state an open-shell singlet?"""
        return self.ctx.gs_mult == 1

    def submit_nat_orb(self):
        self.report("Submitting natural pop analysis")

        builder = GaussianNatOrbWorkChain.get_builder()
        builder.gaussian_code = self.inputs.gaussian_code
        builder.parent_calc_folder = self.ctx.gs_scf_remote_folder
        builder.parent_calc_params = self.ctx.gs_out_params
        if "options" in self.inputs:
            builder.options = self.inputs.options

        submitted_node = self.submit(builder)
        submitted_node.description = "natural orbitals pop"
        self.to_context(natorb=submitted_node)

    def inspect_nat_orb(self):
        if not common_utils.check_if_calc_ok(self, self.ctx.natorb):
            return self.exit_codes.ERROR_TERMINATION

        self.out("gs_natorb_params", self.ctx.natorb.outputs.natorb_proc_parameters)

        return engine.ExitCode(0)

    def finalize(self):
        self.report("Finalizing...")
