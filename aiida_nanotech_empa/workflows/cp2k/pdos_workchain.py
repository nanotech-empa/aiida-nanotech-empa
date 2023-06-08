import copy

import numpy as np
from aiida import engine, orm, plugins

from ...utils import common_utils
from . import cp2k_utils

Cp2kDiagWorkChain = plugins.WorkflowFactory("nanotech_empa.cp2k.diag")
OverlapCalculation = plugins.CalculationFactory("nanotech_empa.overlap")


class Cp2kPdosWorkChain(engine.WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("cp2k_code", valid_type=orm.Code)
        spec.input("slabsys_structure", valid_type=orm.StructureData)
        spec.input("mol_structure", valid_type=orm.StructureData)
        spec.input("pdos_lists", valid_type=orm.List)
        spec.input("parent_calc_folder", valid_type=orm.RemoteData, required=False)
        spec.input(
            "protocol",
            valid_type=orm.Str,
            default=lambda: orm.Str("standard"),
            required=False,
            help="Protocol supported by the Cp2kDiagWorkChain workchain.",
        )
        spec.input("dft_params", valid_type=orm.Dict)
        spec.input("overlap_code", valid_type=orm.Code)
        spec.input("overlap_params", valid_type=orm.Dict)
        spec.input_namespace(
            "options",
            valid_type=int,
            non_db=True,
            help="Define options for the cacluations: walltime, memory, CPUs, etc.",
        )

        spec.outline(
            cls.setup,
            cls.run_diags,
            cls.run_overlap,
            cls.finalize,
        )

        spec.outputs.dynamic = True

        spec.exit_code(
            390,
            "ERROR_TERMINATION",
            message="One or more steps of the work chain failed.",
        )

    def setup(self):
        self.report("Setting up workchain")

        self.ctx.n_slab_atoms = len(self.inputs.slabsys_structure.sites)
        self.ctx.slab_options = self.inputs.options["slab"]
        self.ctx.mol_options = self.inputs.options["molecule"]
        emax = float(self.inputs.overlap_params.get_dict()["--emax1"])
        nlumo = int(self.inputs.overlap_params.get_dict()["--nlumo2"])

        # Set up slab DFT parameters.
        self.ctx.slab_dft_params = copy.deepcopy(self.inputs.dft_params.get_dict())
        self.ctx.slab_dft_params["added_mos"] = np.max(
            [100, int(1.2 * self.ctx.n_slab_atoms * emax / 5.0)]
        )

        # Set up mol DFT parameters.
        self.ctx.mol_dft_params = copy.deepcopy(self.ctx.slab_dft_params)
        self.ctx.mol_dft_params["added_mos"] = nlumo + 2

        # Force same cutoff for molecule and slab.
        self.ctx.mol_dft_params["cutoff"] = cp2k_utils.get_cutoff(
            self.inputs.slabsys_structure
        )
        self.ctx.mol_dft_params[
            "elpa_switch"
        ] = False  # Elpa can cause problems with small systems

        slab_atoms = self.inputs.slabsys_structure.get_ase()
        mol_atoms = self.inputs.mol_structure.get_ase()

        if self.ctx.mol_dft_params["uks"]:
            mol_spin_up = []
            mol_spin_dw = []
            mol_at_tuples = [
                (e, *np.round(p, 2))
                for e, p in zip(mol_atoms.get_chemical_symbols(), mol_atoms.positions)
            ]

            for i_up in self.ctx.mol_dft_params["spin_up_guess"]:
                at = slab_atoms[i_up]
                at_tup = (at.symbol, *np.round(at.position, 2))
                if at_tup in mol_at_tuples:
                    mol_spin_up.append(mol_at_tuples.index(at_tup))

            for i_dw in self.ctx.mol_dft_params["spin_dw_guess"]:
                at = slab_atoms[i_dw]
                at_tup = (at.symbol, *np.round(at.position, 2))
                if at_tup in mol_at_tuples:
                    mol_spin_dw.append(mol_at_tuples.index(at_tup))

            self.ctx.mol_dft_params["spin_up_guess"] = mol_spin_up
            self.ctx.mol_dft_params["spin_dw_guess"] = mol_spin_dw

    def run_diags(self):
        # Slab part.
        self.report("Running Diag Workchain for slab")

        builder = Cp2kDiagWorkChain.get_builder()
        builder.cp2k_code = self.inputs.cp2k_code
        builder.structure = self.inputs.slabsys_structure
        builder.protocol = self.inputs.protocol
        builder.dft_params = orm.Dict(self.ctx.slab_dft_params)
        builder.settings = orm.Dict({"additional_retrieve_list": ["*.pdos"]})
        builder.options = orm.Dict(self.ctx.slab_options)

        # Restart WFN.
        if "parent_calc_folder" in self.inputs:
            builder.parent_calc_folder = self.inputs.parent_calc_folder
        # PDOS.
        if self.inputs.pdos_lists is not None:
            builder.pdos_lists = orm.List([pdos[0] for pdos in self.inputs.pdos_lists])

        slab_future = self.submit(builder)
        self.to_context(slab_diag_scf=slab_future)
        # End slab part.

        # Molecule part.
        self.report("Running Diag Workchain for molecule")

        builder = Cp2kDiagWorkChain.get_builder()
        builder.cp2k_code = self.inputs.cp2k_code
        builder.structure = self.inputs.mol_structure
        builder.protocol = self.inputs.protocol
        builder.dft_params = orm.Dict(self.ctx.mol_dft_params)
        builder.options = orm.Dict(self.ctx.mol_options)

        mol_future = self.submit(builder)
        self.to_context(mol_diag_scf=mol_future)
        # End molecule part.

    def run_overlap(self):
        for calculation in [self.ctx.slab_diag_scf, self.ctx.mol_diag_scf]:
            if not common_utils.check_if_calc_ok(self, calculation):
                return self.exit_codes.ERROR_TERMINATION
        self.report("Running overlap")

        inputs = {}
        inputs["metadata"] = {}
        inputs["metadata"]["label"] = "overlap"
        inputs["code"] = self.inputs.overlap_code
        inputs["parameters"] = self.inputs.overlap_params
        inputs["parent_slab_folder"] = self.ctx.slab_diag_scf.outputs.remote_folder
        inputs["parent_mol_folder"] = self.ctx.mol_diag_scf.outputs.remote_folder

        n_machines = 4 if self.ctx.n_slab_atoms < 2000 else 8

        inputs["metadata"]["options"] = {
            "resources": {"num_machines": n_machines},
            "max_wallclock_seconds": 86400,
        }
        settings = orm.Dict({"additional_retrieve_list": ["overlap.npz"]})
        inputs["settings"] = settings
        future = self.submit(OverlapCalculation, **inputs)
        return engine.ToContext(overlap=future)

    def finalize(self):
        if "overlap.npz" not in [
            obj.name for obj in self.ctx.overlap.outputs.retrieved.list_objects()
        ]:
            self.report("Overlap calculation did not finish correctly")
            return self.exit_codes.ERROR_TERMINATION
        self.out("slab_retrieved", self.ctx.slab_diag_scf.outputs.retrieved)

        # Add the workchain uuid to the input structure extras.
        common_utils.add_extras(
            self.inputs.slabsys_structure, "surfaces", self.node.uuid
        )
        self.report("Work chain is finished")
