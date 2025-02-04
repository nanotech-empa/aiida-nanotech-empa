from copy import deepcopy

import numpy as np
from aiida import engine, orm, plugins
from aiida_cp2k.utils import merge_trajectory_data_unique

# from ...utils import common_utils
from . import cp2k_utils

Cp2kBaseWorkChain = plugins.WorkflowFactory("cp2k.base")
# Cp2kRefTrajWorkChain = plugins.WorkflowFactory("cp2k.reftraj")
TrajectoryData = plugins.DataFactory("array.trajectory")


def last_reftraj_wc(input_trajectory):
    """Identify latest workchain that run the same input trajectory."""

    # Query for WorkChainNode that had this node as input and has the label "mylabel"

    qb = orm.QueryBuilder()
    qb.append(
        orm.Node, filters={"uuid": input_trajectory.uuid}, tag="input_node"
    )  # Input node filter
    qb.append(
        orm.WorkChainNode,
        filters={
            "label": "CP2K_RefTraj",
            "attributes.process_state": {"in": ["finished", "excepted", "killed"]},
        },  # Filter for workchain label "mylabel"
        with_incoming="input_node",  # WorkChain must have this node as input
        # project=['id'],  # Only project the PK (id)
        tag="workchain",
    )

    # Sort by the ctime to get the last workchain
    qb.order_by({"workchain": {"ctime": "desc"}})

    # Fetch the result
    result = qb.first()

    if result:
        return result[0]  # Return the PK of the workchain
    else:
        return None  # No workchain found


def retireve_previous_trajectories(reftraj_wc):
    """Identify latest workchain that run teh same input trajectory and retrieve the trajectories from it."""

    trajectories = []
    if reftraj_wc.is_finished_ok:
        trajectories.append(reftraj_wc.outputs.output_trajectory)
    else:
        # check if a merged directory is already available at the beginnign of the previous workchain
        if reftraj_wc.called_descendants[0].label == "merge_trajectory_data_unique":
            trajectories = [reftraj_wc.called_descendants[0].outputs.result]

        base_workchains = [
            wc
            for wc in reftraj_wc.called_descendants
            if wc.process_label == "Cp2kBaseWorkChain"
        ]
        for base_wc in base_workchains:
            # if BaseWorkChain is_finished_ok use the TrajectoryData
            if base_wc.is_finished_ok:
                trajectories.append(base_wc.outputs.output_trajectory)
            # otehrwise use the data from each completed cp2kcalc
            else:
                cp2k_calcs = [
                    calc
                    for calc in base_wc.called_descendants
                    if calc.process_label == "Cp2kCalculation" and calc.is_finished_ok
                ]
                for calc in cp2k_calcs:
                    trajectories.append(calc.outputs.output_trajectory)
    return orm.List(trajectories)


@engine.calcfunction
def merge_trajectories(*trajectories):
    """Merge a list of trajectories into a single one."""

    arrays = {}
    traj_keys = trajectories[0].get_arraynames()
    symbols = trajectories[0].symbols
    traj_keys.remove("steps")
    for key in traj_keys:
        arrays[key] = []
    for trajectory in trajectories:
        for key in traj_keys:
            arrays[key].append(trajectory.get_array(key))

    merged_trajectory = TrajectoryData()
    if "cells" in traj_keys:
        merged_trajectory.set_trajectory(
            symbols,
            np.concatenate(arrays["positions"]),
            cells=np.concatenate(arrays["cells"]),
        )
    else:
        merged_trajectory.set_trajectory(symbols, np.concatenate(arrays["positions"]))
    traj_keys = [key for key in traj_keys if key not in ["cells", "positions"]]
    for key in traj_keys:
        merged_trajectory.set_array(key, np.concatenate(arrays[key]))

    return merged_trajectory


# @engine.calcfunction
def create_batches(trajectory, num_batches, steps_completed):
    """Create lists of consecutive integers. Counting start from 1 for CP2K input. The first list contains only one element."""

    input_list = [i + 1 for i in range(trajectory.get_shape("positions")[0])]
    for i in steps_completed:
        input_list.remove(i)

    if len(input_list) == 0:
        return {}
    # If there are fewer elements than num_batches + 1, return each element as a separate list
    if len(input_list) < num_batches.value + 1:
        return {i: [value] for i, value in enumerate(input_list)}

    # Initialize the batches with the first batch containing only the first element
    batches = [[input_list[0]]]

    # Calculate the number of remaining elements to distribute among other batches
    remaining_elements = input_list[1:]
    total_remaining = len(remaining_elements)

    # Calculate the minimum number of elements each batch must have
    min_elements_per_batch = total_remaining // num_batches.value
    extra_elements = (
        total_remaining % num_batches.value
    )  # Determine how many batches will have an extra element

    start_idx = 0
    for i in range(num_batches.value):
        # If there are extra elements, add one more to this batch
        end_idx = start_idx + min_elements_per_batch + (1 if i < extra_elements else 0)
        batches.append(remaining_elements[start_idx:end_idx])
        start_idx = end_idx

    return dict(enumerate(batches))


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
        spec.input("restart", valid_type=orm.Bool, required=False)
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
            engine.if_(cls.something_to_run)(
                cls.first_structure,  # Run the first SCF to get the initial wavefunction
                cls.run_reftraj_batches,
            ),  # Run the batches of the reftraj simulations
            cls.merge_batches_output,
        )

        spec.outputs.dynamic = True
        spec.output_namespace("structures", valid_type=orm.StructureData)
        spec.output_namespace("details", valid_type=orm.Dict)
        spec.exit_code(390, "ERROR_TERMINATION", message="One or more steps failed")

    def setup(self):
        """Initialize the workchain process."""
        self.report("Inspecting input and setting up things")
        self.ctx.previuos_trajectory = None
        self.ctx.steps_completed = []
        restart = self.inputs.get("restart", None)
        if restart:
            last_wc = last_reftraj_wc(self.inputs.trajectory)
            self.report(f"Restrating from last workchain found: {last_wc}")
            previous_trajectories = retireve_previous_trajectories(last_wc)
            self.ctx.previuos_trajectory = merge_trajectory_data_unique(
                *previous_trajectories
            )
            self.ctx.steps_completed = (
                self.ctx.previuos_trajectory.get_stepids().tolist()
            )

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
        # create batches avoiding steps already completed.
        self.ctx.something_to_run = False
        self.ctx.batches = create_batches(
            self.inputs.trajectory, self.inputs.num_batches, self.ctx.steps_completed
        )
        if len(self.ctx.batches) > 0:
            self.ctx.something_to_run = True
            self.ctx.n_batches = len(self.ctx.batches)
        return engine.ExitCode(0)

    def something_to_run(self):
        """Function that returnns whether to run or not soem batch"""
        return self.ctx.something_to_run

    def first_structure(self):
        """Run scf on the initial geometry."""
        input_dict = deepcopy(self.ctx.input_dict)
        batch = self.ctx.batches[0]

        self.report(f"Running structure {batch[0]} to {batch[-1]} ")

        input_dict["MOTION"]["MD"]["REFTRAJ"]["FIRST_SNAPSHOT"] = batch[0]
        input_dict["MOTION"]["MD"]["REFTRAJ"]["LAST_SNAPSHOT"] = batch[-1]

        # create the input for the reftraj workchain
        builder = Cp2kBaseWorkChain.get_builder()
        builder.cp2k.structure = orm.StructureData(ase=self.ctx.structure_with_tags)
        builder.cp2k.trajectory = self.inputs.trajectory
        builder.cp2k.code = self.inputs.code
        builder.cp2k.file = self.ctx.files
        if "parent_calc_folder" in self.inputs:
            builder.cp2k.parent_calc_folder = self.inputs.parent_calc_folder
        builder.cp2k.metadata.options = self.inputs.options
        builder.cp2k.metadata.label = f"structures_{batch[0]}_to_{batch[-1]}"
        builder.cp2k.metadata.options.parser_name = "cp2k_advanced_parser"

        builder.cp2k.parameters = orm.Dict(dict=input_dict)

        future = self.submit(builder)

        key = "reftraj_batch_0"
        self.report(f"Submitted reftraj batch: {key} with pk: {future.pk}")

        self.to_context(**{key: future})

    def run_reftraj_batches(self):
        """Check if all calculations completed and merge trejectories."""
        key0 = "reftraj_batch_0"
        if not getattr(self.ctx, key0).is_finished_ok:
            self.report(f"Batch {key0} failed")
            return self.exit_codes.ERROR_TERMINATION
        for batch in range(1, self.ctx.n_batches):
            first = self.ctx.batches[batch][0]
            last = self.ctx.batches[batch][-1]
            self.report(f"Running structures {first} to {last} ")

            # update the input_dict with the new batch
            input_dict = deepcopy(self.ctx.input_dict)
            input_dict["MOTION"]["MD"]["STEPS"] = 1 + first - last
            input_dict["MOTION"]["MD"]["REFTRAJ"]["FIRST_SNAPSHOT"] = first
            input_dict["MOTION"]["MD"]["REFTRAJ"]["LAST_SNAPSHOT"] = last

            # create the input for the reftraj workchain
            builder = Cp2kBaseWorkChain.get_builder()
            builder.cp2k.structure = orm.StructureData(ase=self.ctx.structure_with_tags)
            builder.cp2k.trajectory = self.inputs.trajectory
            builder.cp2k.code = self.inputs.code
            builder.cp2k.file = self.ctx.files
            builder.cp2k.metadata.options = self.inputs.options
            builder.cp2k.metadata.label = f"structures_{first}_to_{last}"
            builder.cp2k.metadata.options.parser_name = "cp2k_advanced_parser"
            builder.cp2k.parameters = orm.Dict(dict=input_dict)
            builder.cp2k.parent_calc_folder = getattr(
                self.ctx, key0
            ).outputs.remote_folder

            future = self.submit(builder)

            key = f"reftraj_batch_{batch}"
            self.report(f"Submitted reftraj batch: {key} with pk: {future.pk}")

            self.to_context(**{key: future})

    def merge_batches_output(self):
        """Merge the output of the succefull batches only."""

        # merged_traj = []
        # for i_batch in range(self.ctx.n_batches):
        #    merged_traj.extend(self.ctx[f"reftraj_batch_{i_batch}"].outputs.trajectory)

        trajectories_to_merge = []
        for batch in self.ctx.batches:
            key = f"reftraj_batch_{batch}"
            if not getattr(self.ctx, key).is_finished_ok:
                self.report(f"Batch {key} failed")
                return self.exit_codes.ERROR_TERMINATION
            trajectories_to_merge.append(
                getattr(self.ctx, key).outputs.output_trajectory
            )
        if self.ctx.previuos_trajectory is not None:
            trajectories_to_merge.append(self.ctx.previuos_trajectory)
        merged_trajectory = merge_trajectory_data_unique(
            *trajectories_to_merge
        )  # merge_trajectories(*trajectories_to_merge)

        self.out("output_trajectory", merged_trajectory)
        self.report("done")
        return engine.ExitCode(0)
