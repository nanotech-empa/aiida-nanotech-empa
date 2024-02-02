import numpy as np
from aiida import engine, orm, plugins

from ...utils import common_utils
from . import cp2k_utils

# Cp2kBaseWorkChain = plugins.WorkflowFactory("cp2k.base")
Cp2kCalculation = plugins.CalculationFactory("cp2k")
TrajectoryData = plugins.DataFactory("array.trajectory")


# function to check if all ase structures have the same cell
def check_cell(ase_structures, cell_tolerance=1e-4):
    """Check if all the structures in the trajectory have the same cell."""
    cell = ase_structures[0].get_cell()
    for s in ase_structures[1:]:
        if not np.allclose(cell, s.get_cell(), atol=cell_tolerance):
            return False
    return True


# fuction that takes a list of ase structures and returns different leists where in each list teh structures have the same cell and the same formula
def split_structures(ase_structures):
    """Split the structures in the trajectory in different lists based on the cell and the formula."""
    formulas = [s.get_chemical_formula() for s in ase_structures]
    unique_formulas = list(set(formulas))
    split_structures = [
        [s for s, f in zip(ase_structures, formulas) if f == uf]
        for uf in unique_formulas
    ]
    return split_structures


@calcfunction
def create_batches(trajectory, batch_size):
    """Create batches of the trajectory."""
    ase_structures = trajectory.get_ase()
    if not check_cell(ase_structures):
        raise ValueError(
            "The cell of the structures in the trajectory is not the same."
        )
    split_structures = split_structures(ase_structures)
    batches = []
    for ss in split_structures:
        for i in range(0, len(ss), batch_size):
            batches.append(ss[i : i + batch_size])
    return batches


class Cp2kReftrajMdWorkChain(engine.WorkChain):
    """Workflow to run Replica Chain calculations with CP2K."""

    @classmethod
    def define(cls, spec):
        """Define the workflow."""
        super().define(spec)

        # Define the inputs of the workflow.
        spec.input("code", valid_type=orm.Code)
        spec.input("trajectory", valid_type=TrajectoryData)
        spec.input("batch_size", valid_type=orm.Int, default=lambda: orm.Int(10))
        spec.input("parent_calc_folder", valid_type=orm.RemoteData, required=False)
        spec.input("restart_from", valid_type=orm.Str, required=False)
        spec.input(
            "protocol",
            valid_type=orm.Str,
            default=lambda: orm.Str("standard"),
            required=False,
            help="Protocol supported by the Cp2kBaseWorkChain.",
        )
        spec.input("dft_params", valid_type=orm.Dict)
        spec.input("sys_params", valid_type=orm.Dict)
        spec.input(
            "options",
            valid_type=dict,
            non_db=True,
            help="Define options for the cacluations: walltime, memory, CPUs, etc.",
        )

        spec.outline(
            cls.setup,  # create batches, if reordering of structures create indexing
            cls.first_scf,  # Run the first SCF to get the initial wavefunction
            cls.run_reftraj_batches,  # Run the batches of the reftraj simulations
            cls.merge_batches_output,
        )

        spec.outputs.dynamic = True
        spec.output_namespace("structures", valid_type=orm.StructureData)
        spec.output_namespace("details", valid_type=orm.Dict)
        spec.exit_code(390, "ERROR_TERMINATION", message="One geo opt failed")

    def setup(self):
        """Initialize the workchain process."""
        self.report("Inspecting input and setting up things")
        return engine.ExitCode(0)

    def first_scf(self):
        """Run scf on the initial geometry."""

        files, input_dict, structure_with_tags = cp2k_utils.get_dft_inputs(
            self.inputs.dft_params.get_dict(),
            self.ctx.lowest_energy_structure,
            "scf_ot_protocol.yml",
            self.inputs.protocol.value,
        )

        builder = Cp2kBaseWorkChain.get_builder()
        builder.cp2k.structure = orm.StructureData(ase=structure_with_tags)
        builder.cp2k.code = self.inputs.code
        builder.cp2k.file = files
        if "parent_calc_folder" in self.inputs:
            builder.cp2k.parent_calc_folder = self.inputs.parent_calc_folder
        builder.cp2k.metadata.options = self.inputs.options
        builder.cp2k.metadata.label = "scf"
        builder.cp2k.metadata.options.parser_name = "cp2k_advanced_parser"
        input_dict["GLOBAL"]["WALLTIME"] = max(
            600, self.inputs.options["max_wallclock_seconds"] - 600
        )
        cp2k.ctx.input_dict = input_dict
        builder.cp2k.parameters = orm.Dict(input_dict)

        future = self.submit(builder)
        self.report(f"Submitted SCF of the initial geometry: {future.pk}")
        self.ToContext(initial_scf=future)

    def run_reftraj_batches(self):
        for i_batch, trajectory_batch in enumerate(
            create_batches(self.inputs.trajectory, batch_size=self.inputs.batch_size)
        ):
            self.report(f"Running batch of {len(trajectory_batch)} structures")
            # create the input for the reftraj workchain
            builder = Cp2kBaseWorkChain.get_builder()
            builder.cp2k.trajectory = trajectory_batch
            builder.cp2k.code = self.inputs.code
            builder.cp2k.parameters = orm.Dict(dict=self.ctx.input_dict)

            builder.cp2k.parent_calc_folder = self.ctx.initial_scf.outputs.remote_folder
            future = self.submit(Cp2kBaseWorkChain, **reftraj_input)
            self.report(f"Submitted reftraj batch: {i_batch} with pk: {future.pk}")
            key = f"reftraj_batch_{i_batch}"
            self.to_context(**{key: future})

    def merge_batches_output(self):
        """Merge the output of the succefull batches only."""
        merged_traj = []
        for i_batch in range(self.ctx.n_batches):
            merged_traj.extend(self.ctx[f"reftraj_batch_{i_batch}"].outputs.trajectory)
        return merged_traj
