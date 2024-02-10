import numpy as np
from aiida import engine, orm, plugins

from ...utils import common_utils
from . import cp2k_utils

# Cp2kBaseWorkChain = plugins.WorkflowFactory("cp2k.base")
Cp2kCalculation = plugins.CalculationFactory("cp2k")
TrajectoryData = plugins.DataFactory("array.trajectory")


@calcfunction
def create_batches(trajectory, batch_size):
    """Create a list of touples with integers as batches indeces. Counting start from 1 for CP2K input,
    batches start from 2 since the first structure runs separately.
    """
    n_structures = trajectory.get_shape("positions")[1]
    batches = []

    for i_batch in range(2, n_structures + 1, batch_size):
        batches.append((i_batch, min(i_batch + batch_size - 1, n_structures)))

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
            cls.first_structure,  # Run the first SCF to get the initial wavefunction
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

        self.ctx.files, self.ctx.input_dict, self.ctx.structure_with_tags = (
            cp2k_utils.get_dft_inputs(
                self.inputs.dft_params.get_dict(),
                self.inputs.trajectory,
                "md_reftraj_protocol.yml",
                self.inputs.protocol.value,
            )
        )
        return engine.ExitCode(0)

    def first_structure(self):
        """Run scf on the initial geometry."""

        builder = Cp2kBaseWorkChain.get_builder()
        builder.cp2k.structure = orm.StructureData(ase=self.ctx.structure_with_tags)
        builder.cp2k.trajectory = self.inputs.trajectory
        builder.cp2k.code = self.inputs.code
        builder.cp2k.file = self.ctx.files
        if "parent_calc_folder" in self.inputs:
            builder.cp2k.parent_calc_folder = self.inputs.parent_calc_folder
        builder.cp2k.metadata.options = self.inputs.options
        builder.cp2k.metadata.label = "structures_1_to_1"
        builder.cp2k.metadata.options.parser_name = "cp2k_advanced_parser"
        input_dict["GLOBAL"]["WALLTIME"] = max(
            600, self.inputs.options["max_wallclock_seconds"] - 600
        )

        builder.cp2k.parameters = orm.Dict(input_dict)

        future = self.submit(builder)
        self.report(f"Submitted structures 1 to 1: {future.pk}")
        self.ToContext(first_structure=future)

    def run_reftraj_batches(self):
        for i_batch, batch in enumerate(
            create_batches(self.inputs.trajectory, batch_size=self.inputs.batch_size)
        ):
            self.report(f"Running structures {batch[0]} to {batch[1]}")
            # create the input for the reftraj workchain
            builder = Cp2kBaseWorkChain.get_builder()
            builder.cp2k.structure = orm.StructureData(ase=self.ctx.structure_with_tags)
            builder.cp2k.trajectory = self.inputs.trajectory
            builder.cp2k.code = self.inputs.code
            builder.cp2k.parameters = orm.Dict(dict=self.ctx.input_dict)

            builder.cp2k.parent_calc_folder = (
                self.ctx.first_structure.outputs.remote_folder
            )
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
