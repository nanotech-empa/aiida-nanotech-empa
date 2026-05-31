import copy
import tempfile
from pathlib import Path

import numpy as np
from aiida import engine, orm, plugins

from ...utils import common_utils, split_structure
from . import cp2k_utils
from .geo_opt_workchain import validate_on_unhandled_failure

Cp2kDiagWorkChain = plugins.WorkflowFactory("nanotech_empa.cp2k.diag")
OverlapCalculation = plugins.CalculationFactory("nanotech_empa.overlap")


def _cp2k_input_without_aux_basis(remote_folder):
    """Return a CP2K input SinglefileData suitable for cp2k_spm_tools overlap.

    cp2k_spm_tools currently treats every BASIS_SET line in a KIND section as the
    orbital basis.  ADMM inputs also contain BASIS_SET AUX_FIT/RI_AUX lines, which
    would overwrite the orbital basis name during parsing.  The overlap tool only
    needs the orbital basis, so remove auxiliary basis lines from this copy.
    """
    text = remote_folder.creator.base.repository.get_object_content("aiida.inp")
    cleaned_lines = []
    for line in text.splitlines(keepends=True):
        parts = line.split()
        if len(parts) >= 3 and parts[0].upper() == "BASIS_SET":
            if parts[1].upper() in {"AUX_FIT", "RI_AUX"}:
                continue
        cleaned_lines.append(line)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "aiida_overlap.inp"
        input_path.write_text("".join(cleaned_lines))
        return orm.SinglefileData(file=input_path)


class Cp2kPdosWorkChain(engine.WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)

        # Do overlap.
        spec.input("do_overlap", valid_type=orm.Bool, default=lambda: orm.Bool(False))

        # Codes.
        spec.input("cp2k_code", valid_type=orm.Code)
        spec.input("overlap_code", valid_type=orm.Code, required=False, default=None)

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
        spec.input(
            "max_iterations",
            valid_type=orm.Int,
            default=lambda: orm.Int(5),
            required=False,
            help="Maximum number of CP2K restart attempts delegated to cp2k.base.",
        )
        spec.input(
            "clean_workdir",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            required=False,
            help="Clean called CP2K calculation work directories after termination.",
        )
        spec.input(
            "on_unhandled_failure",
            valid_type=orm.Str,
            default=lambda: orm.Str("pause"),
            required=False,
            validator=validate_on_unhandled_failure,
            help="Action for unhandled cp2k.base failures: abort, pause, restart_once, or restart_and_pause.",
        )
        spec.input(
            "pause_on_max_iterations",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(True),
            required=False,
            help="Pause cp2k.base for inspection when restart max_iterations is reached.",
        )

        spec.outline(
            cls.setup,
            cls.run_diags,
            engine.if_(cls.should_run_overlap)(
                cls.run_overlap,
            ),
            cls.finalize,
        )

        spec.outputs.dynamic = True

        spec.exit_code(
            390,
            "ERROR_TERMINATION",
            message="One or more steps of the work chain failed.",
        )

    def set_restart_policy(self, builder):
        builder.max_iterations = self.inputs.max_iterations
        builder.clean_workdir = self.inputs.clean_workdir
        builder.on_unhandled_failure = self.inputs.on_unhandled_failure
        builder.pause_on_max_iterations = self.inputs.pause_on_max_iterations

    def setup(self):
        self.report("Setting up workchain")
        structure_generator = split_structure.split_structure(
            structure=self.inputs.structure,
            fixed_atoms=[],
            magnetization_per_site=(
                self.inputs.dft_params["magnetization_per_site"]
                if "magnetization_per_site" in self.inputs.dft_params
                else None
            ),
            fragments={"molecule": self.inputs.molecule_indices},
        )

        self.ctx.n_slab_atoms = len(self.inputs.structure.sites)
        self.ctx.do_overlap = (
            self.inputs.do_overlap.value and self.inputs.overlap_code is not None
        )
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
        self.ctx.mol_dft_parameters["elpa_switch"] = (
            False  # Elpa can cause problems with small systems
        )
        self.ctx.mol_dft_parameters["magnetization_per_site"] = molecule_info[
            "magnetization_per_site"
        ]

    def run_diags(self):
        # Full system part.
        self.report("Running Diag Workchain for the full system.")
        builder = Cp2kDiagWorkChain.get_builder()
        builder.cp2k_code = self.inputs.cp2k_code
        self.set_restart_policy(builder)
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

        # Fragment part.
        if self.ctx.do_overlap:
            self.report("Running Diag Workchain for the fragment.")
            builder = Cp2kDiagWorkChain.get_builder()
            builder.cp2k_code = self.inputs.cp2k_code
            self.set_restart_policy(builder)
            builder.structure = self.ctx.molecule_structure
            builder.protocol = self.inputs.protocol
            builder.dft_params = orm.Dict(self.ctx.mol_dft_parameters)
            builder.options = orm.Dict(self.inputs.options["molecule"])
            self.to_context(mol_diag_scf=self.submit(builder))

    def should_run_overlap(self):
        return self.ctx.do_overlap

    def run_overlap(self):
        for calculation in [self.ctx.slab_diag_scf, self.ctx.mol_diag_scf]:
            if not common_utils.check_if_calc_ok(self, calculation):
                return self.exit_codes.ERROR_TERMINATION
        self.report("Running overlap")
        builder = OverlapCalculation.get_builder()
        builder.code = self.inputs.overlap_code
        overlap_params = self.inputs.overlap_params.get_dict()
        basis_file = cp2k_utils.get_dft_file_names(self.ctx.dft_parameters)[
            "basis_set_file_names"
        ][0]
        overlap_params["--basis_set_file1"] = f"parent_slab_folder/{basis_file}"
        overlap_params["--basis_set_file2"] = f"parent_mol_folder/{basis_file}"
        builder.parameters = orm.Dict(overlap_params)
        builder.parent_slab_folder = self.ctx.slab_diag_scf.outputs.remote_folder
        builder.parent_mol_folder = self.ctx.mol_diag_scf.outputs.remote_folder
        builder.slab_cp2k_input = _cp2k_input_without_aux_basis(
            self.ctx.slab_diag_scf.outputs.remote_folder
        )
        builder.mol_cp2k_input = _cp2k_input_without_aux_basis(
            self.ctx.mol_diag_scf.outputs.remote_folder
        )

        if self.ctx.n_slab_atoms < 500:
            n_machines = 1
            walltime = 7200
        elif self.ctx.n_slab_atoms < 1000:
            n_machines = 2
            walltime = 14400
        elif self.ctx.n_slab_atoms < 2000:
            n_machines = 4
            walltime = 36000
        else:
            n_machines = 8
            walltime = 86400

        builder.metadata = {
            "label": "overlap",
            "options": {
                "resources": {
                    "num_machines": n_machines,
                    "num_mpiprocs_per_machine": min(
                        36,
                        self.inputs.cp2k_code.computer.get_default_mpiprocs_per_machine(),
                    ),
                    "num_cores_per_mpiproc": 1,
                },
                "max_wallclock_seconds": walltime,
            },
        }

        builder.settings = orm.Dict({"additional_retrieve_list": ["overlap.npz"]})
        future = self.submit(builder)
        return engine.ToContext(overlap=future)

    def finalize(self):
        self.report("Finalizing workchain")
        if self.ctx.do_overlap:
            if "overlap.npz" not in [
                obj.name for obj in self.ctx.overlap.outputs.retrieved.list_objects()
            ]:
                self.report("Overlap calculation did not finish correctly")
                return self.exit_codes.ERROR_TERMINATION
        self.out("slab_retrieved", self.ctx.slab_diag_scf.outputs.retrieved)

        # Add the workchain uuid to the input structure extras.
        common_utils.add_extras(self.inputs.structure, "surfaces", self.node.uuid)
        self.report("Work chain is finished")
