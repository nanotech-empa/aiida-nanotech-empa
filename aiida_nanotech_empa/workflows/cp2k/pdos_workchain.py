import copy

import numpy as np
from aiida import engine, orm, plugins

from ...utils import common_utils, split_structure
from . import cp2k_utils

Cp2kDiagWorkChain = plugins.WorkflowFactory("nanotech_empa.cp2k.diag")
OverlapCalculation = plugins.CalculationFactory("nanotech_empa.overlap")


class Cp2kPdosWorkChain(engine.WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)

        # Codes.
        spec.input("cp2k_code", valid_type=orm.Code)
        spec.input("overlap_code", valid_type=orm.Code)

        # Structures.
        spec.input(
            "structure",
            valid_type=orm.StructureData,
            help="Coordinates of the whole system.",
        )
        spec.input("molecule_indices", valid_type=orm.List)

        spec.input("pdos_lists", valid_type=orm.List)

        # Numerical parameters.
        spec.input(
            "protocol",
            valid_type=orm.Str,
            default=lambda: orm.Str("standard"),
            required=False,
            help="Protocol supported by the Cp2kDiagWorkChain workchain.",
        )
        spec.input("dft_params", valid_type=orm.Dict)
        spec.input("overlap_params", valid_type=orm.Dict)

        # High-level things.
        spec.input("parent_calc_folder", valid_type=orm.RemoteData, required=False)
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
        structure_generator = split_structure.split_structure(
            structure=self.inputs.structure,
            fixed_atoms=[],
            magnetization_per_site=self.inputs.dft_params["magnetization_per_site"]
            if "magnetization_per_site" in self.inputs.dft_params
            else None,
            fragments={"molecule": self.inputs.molecule_indices},
        )

        self.ctx.n_slab_atoms = len(self.inputs.structure.sites)
        emax = float(self.inputs.overlap_params.get_dict()["--emax1"])
        nlumo = int(self.inputs.overlap_params.get_dict()["--nlumo2"])

        dft_parameters = self.inputs.dft_params.get_dict()
        charges = dft_parameters.pop("charges")
        multiplicities = dft_parameters.pop("multiplicities", {})

        # Set up DFT parameters of the whole system.
        slab_info = next(structure_generator)
        self.ctx.structure = slab_info["structure"]
        self.ctx.dft_parameters = copy.deepcopy(dft_parameters)
        self.ctx.dft_parameters["charge"] = charges["all"]
        if "all" in multiplicities:
            self.ctx.dft_parameters["multiplicity"] = multiplicities["all"]
        self.ctx.dft_parameters["added_mos"] = np.max(
            [20, int(1.2 * self.ctx.n_slab_atoms * emax / 5.0)]
        )

        # Use the same cutoff for molecule and slab.
        self.ctx.dft_parameters["cutoff"] = cp2k_utils.get_cutoff(self.ctx.structure)

        # Set up molecular DFT parameters.
        molecule_info = next(structure_generator)
        self.ctx.molecule_structure = molecule_info["structure"]
        self.ctx.mol_dft_parameters = copy.deepcopy(self.ctx.dft_parameters)
        self.ctx.mol_dft_parameters["charge"] = charges["molecule"]
        if "molecule" in multiplicities:
            self.ctx.mol_dft_parameters["multiplicity"] = multiplicities["molecule"]
        self.ctx.mol_dft_parameters["added_mos"] = nlumo + 2
        self.ctx.mol_dft_parameters[
            "elpa_switch"
        ] = False  # Elpa can cause problems with small systems
        self.ctx.mol_dft_parameters["magnetization_per_site"] = molecule_info[
            "magnetization_per_site"
        ]

    def run_diags(self):
        # Slab part.
        self.report("Running Diag Workchain for slab")
        builder = Cp2kDiagWorkChain.get_builder()
        builder.cp2k_code = self.inputs.cp2k_code
        builder.structure = self.ctx.structure
        builder.protocol = self.inputs.protocol
        builder.dft_params = orm.Dict(self.ctx.dft_parameters)
        builder.settings = orm.Dict({"additional_retrieve_list": ["*.pdos"]})
        builder.options = orm.Dict(self.inputs.options["slab"])

        # Restart WFN.
        if "parent_calc_folder" in self.inputs:
            builder.parent_calc_folder = self.inputs.parent_calc_folder

        # PDOS.
        if self.inputs.pdos_lists is not None:
            builder.pdos_lists = orm.List([pdos[0] for pdos in self.inputs.pdos_lists])

        self.to_context(slab_diag_scf=self.submit(builder))

        # Molecule part.
        self.report("Running Diag Workchain for molecule")
        builder = Cp2kDiagWorkChain.get_builder()
        builder.cp2k_code = self.inputs.cp2k_code
        builder.structure = self.ctx.molecule_structure
        builder.protocol = self.inputs.protocol
        builder.dft_params = orm.Dict(self.ctx.mol_dft_parameters)
        builder.options = orm.Dict(self.inputs.options["molecule"])
        self.to_context(mol_diag_scf=self.submit(builder))

    def run_overlap(self):
        for calculation in [self.ctx.slab_diag_scf, self.ctx.mol_diag_scf]:
            if not common_utils.check_if_calc_ok(self, calculation):
                return self.exit_codes.ERROR_TERMINATION
        self.report("Running overlap")
        builder = OverlapCalculation.get_builder()
        builder.code = self.inputs.overlap_code
        builder.parameters = self.inputs.overlap_params
        builder.parent_slab_folder = self.ctx.slab_diag_scf.outputs.remote_folder
        builder.parent_mol_folder = self.ctx.mol_diag_scf.outputs.remote_folder

        n_machines = 4 if self.ctx.n_slab_atoms < 2000 else 8

        builder.metadata = {
            "label": "overlap",
            "options": {
                "resources": {"num_machines": n_machines},
                "max_wallclock_seconds": 86400,
            },
        }

        builder.settings = orm.Dict({"additional_retrieve_list": ["overlap.npz"]})
        future = self.submit(builder)
        return engine.ToContext(overlap=future)

    def finalize(self):
        if "overlap.npz" not in [
            obj.name for obj in self.ctx.overlap.outputs.retrieved.list_objects()
        ]:
            self.report("Overlap calculation did not finish correctly")
            return self.exit_codes.ERROR_TERMINATION
        self.out("slab_retrieved", self.ctx.slab_diag_scf.outputs.retrieved)

        # Add the workchain uuid to the input structure extras.
        common_utils.add_extras(self.inputs.structure, "surfaces", self.node.uuid)
        self.report("Work chain is finished")
