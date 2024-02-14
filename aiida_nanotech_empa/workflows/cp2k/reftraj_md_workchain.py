from copy import deepcopy

import numpy as np
from aiida import engine, orm, plugins

from ...utils import common_utils
from . import cp2k_utils

Cp2kBaseWorkChain = plugins.WorkflowFactory("cp2k.base")
# Cp2kRefTrajWorkChain = plugins.WorkflowFactory("cp2k.reftraj")
TrajectoryData = plugins.DataFactory("array.trajectory")


@engine.calcfunction
def create_batches(trajectory, num_batches, steps_completed):
    """Create lists of consecutive integers. Counting start from 1 for CP2K input."""
    lst = [i + 1 for i in range(trajectory.get_shape("positions")[0])]
    for i in steps_completed:
        lst.remove(i)
    max_batch_size = int(len(lst) / num_batches)
    consecutive_lists = []
    current_list = []
    for num in lst:
        if not current_list or num == current_list[-1] + 1:
            current_list.append(num)
        else:
            consecutive_lists.append(current_list)
            current_list = [num]
        if len(current_list) == max_batch_size:
            consecutive_lists.append(current_list)
            current_list = []
    if current_list:
        consecutive_lists.append(current_list)
    return orm.List(consecutive_lists)


class Cp2kRefTrajWorkChain(engine.WorkChain):
    """Workflow to run Replica Chain calculations with CP2K."""

    @classmethod
    def define(cls, spec):
        """Define the workflow."""
        super().define(spec)

        # Define the inputs of the workflow.
        spec.input("code", valid_type=orm.Code)
        # spec.input("structure", valid_type=orm.StructureData)
        spec.input("trajectory", valid_type=TrajectoryData)
        spec.input("num_batches", valid_type=orm.Int, default=lambda: orm.Int(10))
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
        spec.exit_code(390, "ERROR_TERMINATION", message="One or more steps failed")

    def setup(self):
        """Initialize the workchain process."""
        self.report("Inspecting input and setting up things")

        (
            self.ctx.files,
            self.ctx.input_dict,
            self.ctx.structure_with_tags,
        ) = cp2k_utils.get_dft_inputs(
            self.inputs.dft_params.get_dict(),
            self.inputs.trajectory,
            "md_reftraj_protocol.yml",
            self.inputs.protocol.value,
        )
        self.ctx.input_dict["GLOBAL"]["WALLTIME"] = max(
            600, self.inputs.options["max_wallclock_seconds"] - 600
        )
        self.ctx.steps_completed = []
        # create batches avoiding steps already completed.
        self.ctx.batches = create_batches(
            self.inputs.trajectory, self.inputs.num_batches, self.ctx.steps_completed
        ).get_list()
        return engine.ExitCode(0)

    def first_structure(self):
        """Run scf on the initial geometry."""
        input_dict = deepcopy(self.ctx.input_dict)
        batches = self.ctx.batches
        first_snapshot = batches[0].pop(0)
        self.ctx.batches = batches

        self.report(f"Running structure {first_snapshot} to {first_snapshot} ")

        input_dict["MOTION"]["MD"]["REFTRAJ"]["FIRST_SNAPSHOT"] = first_snapshot
        input_dict["MOTION"]["MD"]["REFTRAJ"]["LAST_SNAPSHOT"] = first_snapshot

        # create the input for the reftraj workchain
        builder = Cp2kBaseWorkChain.get_builder()
        builder.cp2k.structure = orm.StructureData(ase=self.ctx.structure_with_tags)
        builder.cp2k.trajectory = self.inputs.trajectory
        builder.cp2k.code = self.inputs.code
        builder.cp2k.file = self.ctx.files
        if "parent_calc_folder" in self.inputs:
            builder.cp2k.parent_calc_folder = self.inputs.parent_calc_folder
        builder.cp2k.metadata.options = self.inputs.options
        builder.cp2k.metadata.label = f"structures_{first_snapshot}_to_{first_snapshot}"
        builder.cp2k.metadata.options.parser_name = "cp2k_advanced_parser"

        builder.cp2k.parameters = orm.Dict(dict=input_dict)

        future = self.submit(builder)
        self.report(
            f"Submitted structures {first_snapshot} to {first_snapshot}: {future.pk}"
        )
        self.to_context(first_structure=future)

    def run_reftraj_batches(self):
        self.report(f"Running the reftraj batches {self.ctx.batches} ")
        if not self.ctx.first_structure.is_finished_ok:
            return self.exit_codes.ERROR_TERMINATION
        for batch in self.ctx.batches:
            self.report(f"Running structures {batch[0]} to {batch[-1]} ")

            # update the input_dict with the new batch
            input_dict = deepcopy(self.ctx.input_dict)
            input_dict["MOTION"]["MD"]["STEPS"] = 1 + batch[0] - batch[-1]
            input_dict["MOTION"]["MD"]["REFTRAJ"]["FIRST_SNAPSHOT"] = batch[0]
            input_dict["MOTION"]["MD"]["REFTRAJ"]["LAST_SNAPSHOT"] = batch[-1]

            # create the input for the reftraj workchain
            builder = Cp2kBaseWorkChain.get_builder()
            builder.cp2k.structure = orm.StructureData(ase=self.ctx.structure_with_tags)
            builder.cp2k.trajectory = self.inputs.trajectory
            builder.cp2k.code = self.inputs.code
            builder.cp2k.file = self.ctx.files
            builder.cp2k.metadata.options = self.inputs.options
            builder.cp2k.metadata.label = f"structures_{batch[0]}_to_{batch[-1]}"
            builder.cp2k.metadata.options.parser_name = "cp2k_advanced_parser"
            builder.cp2k.parameters = orm.Dict(dict=input_dict)
            builder.cp2k.parent_calc_folder = (
                self.ctx.first_structure.outputs.remote_folder
            )

            future = self.submit(builder)

            key = f"reftraj_batch_{batch[0]}_to_{batch[-1]}"
            self.report(f"Submitted reftraj batch: {key} with pk: {future.pk}")

            self.to_context(**{key: future})

    def merge_batches_output(self):
        """Merge the output of the succefull batches only."""

        # merged_traj = []
        # for i_batch in range(self.ctx.n_batches):
        #    merged_traj.extend(self.ctx[f"reftraj_batch_{i_batch}"].outputs.trajectory)
        positions=[self.ctx.first_structure.outputs.output_trajectory.get_array('positions')]
        cells=[self.ctx.first_structure.outputs.output_trajectory.get_array('cells')]
        forces=[self.ctx.first_structure.outputs.output_trajectory.get_array('forces')]
        for batch in self.ctx.batches:
            key = f"reftraj_batch_{batch[0]}_to_{batch[-1]}"
            if not getattr(self.ctx, key).is_finished_ok:
                self.report(f"Batch {key} failed")
                return self.exit_codes.ERROR_TERMINATION
            positions.append(getattr(self.ctx, key).outputs.output_trajectory.get_array('positions'))
            cells.append(getattr(self.ctx, key).outputs.output_trajectory.get_array('cells'))
            forces.append(getattr(self.ctx, key).outputs.output_trajectory.get_array('forces'))
        
        positions=np.concatenate(positions)
        cells=np.concatenate(cells)
        forces=np.concatenate(forces)
        symbols = self.ctx.first_structure.outputs.output_trajectory.symbols
        output_trajectory = TrajectoryData()   
        output_trajectory.set_trajectory(symbols, positions, cells=cells) 
        self.out('output_trajectory', output_trajectory)
        self.report("done")
        return engine.ExitCode(0)
