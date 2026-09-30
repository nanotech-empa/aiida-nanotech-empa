import math
import pathlib
import re

from aiida import engine, orm
from aiida_cp2k.calculations import Cp2kCalculation

from ...utils import common_utils
from . import cp2k_utils

ALLOWED_PROTOCOLS = ["scf_ot_no_wfn"]


def find_multiples_of_ngpus(ngpus, n, max_N):
    """Return MPI tasks per node divisible by the GPU count.

    The unused node-count argument preserves the draft helper's call signature.
    """
    return list(range(ngpus, max_N + 1, ngpus))


@engine.calcfunction
def analyze_speedup(time_dict):
    """
    Analyzes computational times to find the minimum time per nnodes and
    determines which nnodes cases have speedup efficiency closest to 60% and 50%.

    Parameters:
    time_dict (dict): Dictionary where keys are 'nnodes_ntasks_nthreads' strings,
                      and values are computational times (floats).

    Returns:
    tuple: A tuple containing:
           - min_times_per_nnodes (dict): Minimum time per nnodes.
           - closest_to_60 (int): nnodes value with speedup efficiency closest to 60%.
           - closest_to_50 (int): nnodes value with speedup efficiency closest to 50%.
    """
    from collections import defaultdict

    # Initialize a dictionary to store times per nnodes
    times_per_nnodes = defaultdict(list)

    # Extract nnodes and collect times
    for key, time_and_id in time_dict.items():
        # Split the key to get nnodes, ntasks, nthreads
        nnodes_str, ntasks_str, nthreads_str = key.split("_")
        nnodes = int(nnodes_str)
        time = time_and_id[0]
        # Collect time for each nnodes
        if isinstance(time, (int, float)) and math.isfinite(time) and time > 0:
            times_per_nnodes[nnodes].append(time_and_id)

    # Find the minimum time for each nnodes
    min_times_per_nnodes = {}
    for nnodes, times_and_ids in sorted(times_per_nnodes.items()):
        min_time_and_id = min(times_and_ids, key=lambda x: x[0])
        min_times_per_nnodes[nnodes] = min_time_and_id

    # Sort nnodes to find the lowest nnodes (reference)
    sorted_nnodes = sorted(min_times_per_nnodes.keys())
    if not sorted_nnodes:
        return orm.Dict(
            dict={
                "summary": "No successful benchmark timings. Inspect the child calculations.",
                "closest_to_60": None,
                "closest_to_50": None,
                "min_times_per_nnodes": {},
            }
        )
    Nmin = sorted_nnodes[0]
    time_Nmin = min_times_per_nnodes[Nmin][0]

    # Calculate speedup efficiencies
    speedup_efficiencies = {}
    for N, time_N in min_times_per_nnodes.items():
        actual_speedup = time_Nmin / time_N[0]
        ideal_speedup = N / Nmin
        speedup_efficiency = actual_speedup / ideal_speedup  # Should be between 0 and 1
        speedup_efficiencies[N] = speedup_efficiency

    # Find nnodes closest to 60% and 50% speedup efficiency
    target_efficiencies = [0.6, 0.5]
    closest_nnodes = {}

    for target in target_efficiencies:
        closest_nnodes[target] = None
        min_diff = float("inf")
        for N, efficiency in speedup_efficiencies.items():
            diff = abs(efficiency - target)
            if diff < min_diff:
                min_diff = diff
                closest_nnodes[target] = N

    closest_to_60 = closest_nnodes[0.6]
    closest_to_50 = closest_nnodes[0.5]
    summary = "Minimum times per nnodes:\n"
    for nnodes, time_and_id in min_times_per_nnodes.items():
        summary += (
            f"nnodes: {nnodes}, min_time: {time_and_id[0]}, job_id: {time_and_id[1]}\n"
        )
    summary += f"\nClosest to 60% speedup: nnodes = {closest_to_60}"
    summary += f"\nClosest to 50% speedup: nnodes = {closest_to_50}"

    return orm.Dict(
        dict={
            "summary": summary,
            "closest_to_60": closest_to_60,
            "closest_to_50": closest_to_50,
            "min_times_per_nnodes": min_times_per_nnodes,
        }
    )


@engine.calcfunction
def get_timing_from_FolderData(folder_node=None):
    """
    Parses the 'aiida.out' file contained in the FolderData node with the given pk.
    Returns the sum of the times found in the first occurrence of ' 3 OT CG' and '4 OT LS'.

    Parameters:
    pk (int): The primary key of the FolderData node.

    Returns:
    float: The sum of the two extracted times.
    """
    # Load the FolderData node
    if folder_node is None:
        return orm.Str("FAILED")

    # Check if 'aiida.out' exists in the FolderData
    if "aiida.out" not in folder_node.list_object_names():
        return orm.Str("FAILED")

    # Open 'aiida.out' and read its contents
    with folder_node.open("aiida.out", "r") as f:
        lines = f.readlines()

    time_3_ot_cg = None
    time_4_ot_ls = None

    # Regular expression to extract time (assuming it's a floating-point number)
    time_pattern = re.compile(r"\b(\d+\.\d+)\b")

    for line in lines:
        # Find the first occurrence of ' 3 OT CG' and extract the time
        if time_3_ot_cg is None and " 3 OT CG" in line:
            time_matches = time_pattern.findall(line)
            if time_matches:
                time_3_ot_cg = float(time_matches[0])
            else:
                return orm.Str("FAILED")

        # Find the first occurrence of '4 OT LS' and extract the time
        if time_4_ot_ls is None and "4 OT LS" in line:
            time_matches = time_pattern.findall(line)
            if time_matches:
                time_4_ot_ls = float(time_matches[0])
            else:
                return orm.Str("FAILED")

        # Break the loop if both times have been found
        if time_3_ot_cg is not None and time_4_ot_ls is not None:
            break

    if time_3_ot_cg is None:
        return orm.Str("FAILED")
        # raise ValueError("Could not find ' 3 OT CG' in 'aiida.out'")
    if time_4_ot_ls is None:
        return orm.Str("FAILED")
        # raise ValueError("Could not find '4 OT LS' in 'aiida.out'")

    # Return the sum of the two times
    total_time = time_3_ot_cg + time_4_ot_ls
    if not math.isfinite(total_time) or total_time <= 0:
        return orm.Str("FAILED")
    return orm.Float(total_time)


class Cp2kBenchmarkWorkChain(engine.WorkChain):
    @staticmethod
    def resource_grid(inputs):
        """Select MPI-task multiples of the GPU count within the CPU capacity."""
        capacity = inputs["code"].computer.get_default_mpiprocs_per_machine()
        tasks_per_node = (
            inputs["list_tasks_per_node"].get_list()
            if "list_tasks_per_node" in inputs
            else list(
                range(
                    inputs["ngpus"].value,
                    inputs["max_tasks_per_node"].value + 1,
                    inputs["ngpus"].value,
                )
            )
        )
        return [
            (nodes, tasks, threads)
            for nodes in inputs["list_nodes"]
            for tasks in tasks_per_node
            for threads in inputs["list_threads_per_task"]
            if tasks * threads <= capacity
        ]

    @classmethod
    def validate_inputs(cls, inputs, _):
        if inputs["protocol"].value not in ALLOWED_PROTOCOLS:
            return "Unknown benchmark protocol."
        for name in ("list_nodes", "list_threads_per_task", "list_tasks_per_node"):
            if name not in inputs:
                continue
            values = inputs[name].get_list()
            if not values or any(
                type(value) is not int or value <= 0 for value in values
            ):
                return f"{name} must contain positive integers."
            if len(set(values)) != len(values):
                return f"{name} must not contain duplicates."
        for name in ("ngpus", "max_tasks_per_node", "wallclock", "cutoff"):
            if name in inputs and inputs[name].value <= 0:
                return f"{name} must be positive."
        if "list_tasks_per_node" in inputs and any(
            tasks % inputs["ngpus"].value for tasks in inputs["list_tasks_per_node"]
        ):
            return "MPI tasks per node must be multiples of GPUs per node."
        if inputs["multiplicity"].value < 0:
            return "multiplicity must be non-negative (0 retains the protocol default)."
        code = inputs["code"]
        if code.default_calc_job_plugin != "cp2k":
            return "Select a CP2K code."
        if (
            code.computer is None
            or not code.computer.get_default_mpiprocs_per_machine()
        ):
            return "Configure the computer's default MPI processes per machine (used as the CPU capacity)."
        if not cls.resource_grid(inputs):
            return (
                "No resource combinations satisfy the GPU and CPU-capacity constraints."
            )

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("code", valid_type=orm.Code)
        spec.input("structure", valid_type=orm.StructureData)

        spec.input(
            "protocol",
            valid_type=orm.Str,
            default=lambda: orm.Str("scf_ot_no_wfn"),
            required=False,
            help="Either 'scf_ot_no_wfn', ",
        )
        spec.input(
            "cutoff",
            valid_type=orm.Int,
            required=False,
            help="Cutoff to be used in the benchmark.",
        )
        spec.input(
            "multiplicity",
            valid_type=orm.Int,
            default=lambda: orm.Int(0),
            required=False,
            help="Multiplicity",
        )
        spec.input(
            "wallclock",
            valid_type=orm.Int,
            required=False,
            default=lambda: orm.Int(600),
        )
        spec.input(
            "list_nodes",
            valid_type=orm.List,
            default=lambda: orm.List(list=list(range(1, 11))),
            required=True,
            help="List of #nodes to be used in the benchmark.",
        )
        spec.input(
            "ngpus",
            valid_type=orm.Int,
            default=lambda: orm.Int(4),
            required=True,
            help="Number of GPUs per node.",
        )
        spec.input(
            "max_tasks_per_node",
            valid_type=orm.Int,
            default=lambda: orm.Int(16),
            required=True,
            help="List of #tasks per node to be used in the benchmark.",
        )
        spec.input(
            "list_tasks_per_node",
            valid_type=orm.List,
            required=False,
            help="Explicit MPI task counts per node, each a multiple of ngpus. Overrides max_tasks_per_node.",
        )
        spec.input(
            "list_threads_per_task",
            valid_type=orm.List,
            default=lambda: orm.List(list=[2, 4, 6, 8]),
            required=True,
            help="List of #threads per task to be used in the benchmark.",
        )

        spec.outline(
            cls.setup,
            cls.submit_calculations,
            cls.finalize,
        )
        spec.inputs.validator = cls.validate_inputs
        spec.outputs.dynamic = True

        spec.exit_code(
            381,
            "ERROR_CONVERGENCE1",
            message="SCF of the first step did not converge.",
        )
        spec.exit_code(
            382,
            "ERROR_CONVERGENCE2",
            message="SCF of the second step did not converge.",
        )
        spec.exit_code(
            383,
            "ERROR_NEGATIVE_GAP",
            message="SCF produced a negative gap.",
        )
        spec.exit_code(
            390,
            "ERROR_TERMINATION",
            message="One or more steps of the work chain failed.",
        )

    def setup(self):
        self.report("Inspecting input and setting up things")
        self.ctx.max_tasks = (
            self.inputs.code.computer.get_default_mpiprocs_per_machine()
        )

        files = {
            "basis": orm.SinglefileData(
                file=pathlib.Path(__file__).parent / "data" / "BASIS_MOLOPT"
            ),
            "pseudo": orm.SinglefileData(
                file=pathlib.Path(__file__).parent / "data" / "POTENTIAL"
            ),
        }
        self.ctx.file_uuids = {key: node.store().uuid for key, node in files.items()}
        self.ctx.input_dict = cp2k_utils.load_protocol(
            "benchmarks.yml", self.inputs.protocol.value
        )
        # UKS.
        magnetization_per_site = [0 for i in range(len(self.inputs.structure.sites))]
        multiplicity = self.inputs.multiplicity.value
        if multiplicity:
            # magnetization_per_site = self.ctx.dft_params["magnetization_per_site"]
            self.ctx.input_dict["FORCE_EVAL"]["DFT"]["UKS"] = ".TRUE."
            self.ctx.input_dict["FORCE_EVAL"]["DFT"]["MULTIPLICITY"] = multiplicity

        # Wallclock.
        wallclock = getattr(self.inputs, "wallclock", None)
        if wallclock:
            self.ctx.input_dict["GLOBAL"]["WALLTIME"] = max(600, wallclock.value - 600)

        # Get initial magnetization.
        _, kinds_dict = cp2k_utils.determine_kinds(
            self.inputs.structure, magnetization_per_site
        )

        self.ctx.kinds_section = cp2k_utils.get_kinds_section(
            kinds_dict, protocol="gpw"
        )
        cp2k_utils.dict_merge(self.ctx.input_dict, self.ctx.kinds_section)

        # Overwrite cutoff if given in dft_params.
        cutoff = (
            self.inputs.cutoff.value
            if "cutoff" in self.inputs
            else cp2k_utils.get_cutoff(structure=self.inputs.structure)
        )

        self.ctx.input_dict["FORCE_EVAL"]["DFT"]["MGRID"]["CUTOFF"] = cutoff

        return engine.ExitCode(0)

    def submit_calculations(self):
        input_dict = self.ctx.input_dict

        for nnodes, ntasks, nthreads in self.resource_grid(self.inputs):
            # Prepare the builder.
            builder = Cp2kCalculation.get_builder()
            builder.code = self.inputs.code
            builder.structure = self.inputs.structure
            builder.file = {
                key: orm.load_node(uuid) for key, uuid in self.ctx.file_uuids.items()
            }

            # Options.
            builder.metadata.options = {
                "max_wallclock_seconds": self.inputs.wallclock.value,
                # CalcJob prepend text runs after the configured code's.
                "prepend_text": f"export OMP_NUM_THREADS={nthreads}",
                "resources": {
                    "num_machines": nnodes,
                    "num_mpiprocs_per_machine": ntasks,
                    "num_cores_per_mpiproc": nthreads,
                },
            }

            builder.metadata.options["parser_name"] = "cp2k_advanced_parser"

            builder.parameters = orm.Dict(input_dict)

            submitted_calculation = self.submit(builder)
            self.report(
                f"Submitted nodes {nnodes} tasks per node {ntasks} threads {nthreads}: {submitted_calculation.pk}"
            )
            self.to_context(
                **{f"run_{nnodes}_{ntasks}_{nthreads}": submitted_calculation}
            )

    def finalize(self):
        self.report("Finalizing...")
        result = orm.Dict(dict={})

        for nnodes, ntasks, nthreads in self.resource_grid(self.inputs):
            current_calc = getattr(self.ctx, f"run_{nnodes}_{ntasks}_{nthreads}")
            if not current_calc.is_finished_ok:
                self.report(
                    f"One of the calculations failed: run_{nnodes}_{ntasks}_{nthreads}."
                )
                folder_data = None
            else:
                folder_data = current_calc.outputs.retrieved

            result[f"{nnodes}_{ntasks}_{nthreads}"] = (
                get_timing_from_FolderData(folder_data).value,
                current_calc.get_job_id(),
            )
        result.store()
        self.out("timings", result)
        report = analyze_speedup(result)
        self.out("report", report)
        # Add extras.
        struc = self.inputs.structure
        common_utils.add_extras(struc, "surfaces", self.node.uuid)
        if not report["min_times_per_nnodes"]:
            return self.exit_codes.ERROR_TERMINATION
        return engine.ExitCode(0)
