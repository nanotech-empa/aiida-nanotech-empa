import copy
import pathlib
import re

import numpy as np
from aiida import engine, orm
from aiida_cp2k.calculations import Cp2kCalculation

from . import  cp2k_utils

ALLOWED_PROTOCOLS = ["standard"]

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
    for key, time in time_dict.items():
        # Split the key to get nnodes, ntasks, nthreads
        nnodes_str, ntasks_str, nthreads_str = key.split('_')
        nnodes = int(nnodes_str)
        # Collect time for each nnodes
        times_per_nnodes[nnodes].append(time)

    # Find the minimum time for each nnodes
    min_times_per_nnodes = {}
    for nnodes, times in times_per_nnodes.items():
        min_time = min(times)
        min_times_per_nnodes[nnodes] = min_time

    # Sort nnodes to find the lowest nnodes (reference)
    sorted_nnodes = sorted(min_times_per_nnodes.keys())
    Nmin = sorted_nnodes[0]
    time_Nmin = min_times_per_nnodes[Nmin]

    # Calculate speedup efficiencies
    speedup_efficiencies = {}
    for N, time_N in min_times_per_nnodes.items():
        actual_speedup = time_Nmin / time_N
        ideal_speedup = N / Nmin
        speedup_efficiency = actual_speedup / ideal_speedup  # Should be between 0 and 1
        speedup_efficiencies[N] = speedup_efficiency

    # Find nnodes closest to 60% and 50% speedup efficiency
    target_efficiencies = [0.6, 0.5]
    closest_nnodes = {}

    for target in target_efficiencies:
        closest_nnodes[target] = None
        min_diff = float('inf')
        for N, efficiency in speedup_efficiencies.items():
            diff = abs(efficiency - target)
            if diff < min_diff:
                min_diff = diff
                closest_nnodes[target] = N

    closest_to_60 = closest_nnodes[0.6]
    closest_to_50 = closest_nnodes[0.5]
    summary = "Minimum times per nnodes:\n"
    for nnodes, time in min_times_per_nnodes.items():
        summary+=f"nnodes: {nnodes}, min_time: {time}\n"
        summary+=f"\nClosest to 60% speedup: nnodes = {closest_to_60}"
        summary+=f"\nClosest to 50% speedup: nnodes = {closest_to_50}"

    return orm.Dict(dict={'summary':summary,'closest_to_60':closest_to_60,'closest_to_50':closest_to_50,'min_times_per_nnodes':min_times_per_nnodes})

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
        return orm.Float(100000)
    
    # Check if 'aiida.out' exists in the FolderData
    if 'aiida.out' not in folder_node.list_object_names():
        raise FileNotFoundError(f"'aiida.out' not found in FolderData with uuid {folder_node.uuid}")
    
    # Open 'aiida.out' and read its contents
    with folder_node.open('aiida.out', 'r') as f:
        lines = f.readlines()
    
    time_3_ot_cg = None
    time_4_ot_ls = None
    
    # Regular expression to extract time (assuming it's a floating-point number)
    time_pattern = re.compile(r'\b(\d+\.\d+)\b')
    
    for line in lines:
        # Find the first occurrence of ' 3 OT CG' and extract the time
        if time_3_ot_cg is None and ' 3 OT CG' in line:
            time_matches = time_pattern.findall(line)
            if time_matches:
                time_3_ot_cg = float(time_matches[0])  # Assuming time is the last number
            else:
                raise ValueError(f"No time found in line: '{line.strip()}'")
        
        # Find the first occurrence of '4 OT LS' and extract the time
        if time_4_ot_ls is None and '4 OT LS' in line:
            time_matches = time_pattern.findall(line)
            if time_matches:
                time_4_ot_ls = float(time_matches[0])  # Assuming time is the last number
            else:
                raise ValueError(f"No time found in line: '{line.strip()}'")
        
        # Break the loop if both times have been found
        if time_3_ot_cg is not None and time_4_ot_ls is not None:
            break
    
    if time_3_ot_cg is None:
        raise ValueError("Could not find ' 3 OT CG' in 'aiida.out'")
    if time_4_ot_ls is None:
        raise ValueError("Could not find '4 OT LS' in 'aiida.out'")
    
    # Return the sum of the two times
    total_time = time_3_ot_cg + time_4_ot_ls
    return orm.Float(total_time)   

class Cp2kBenchmarkWorkChain(engine.WorkChain):
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
            default=lambda: orm.List(list=[24]),
            required=True,
            help="List of #nodes to be used in the benchmark.",
        )
        spec.input(
            "list_tasks_per_node",
            valid_type=orm.List,
            default=lambda: orm.List(list=[6]),
            required=True,
            help="List of #tasks per node to be used in the benchmark.",
        )
        spec.input(
            "list_threads_per_task",
            valid_type=orm.List,
            default=lambda: orm.List(list=[2]),
            required=True,
            help="List of #threads per task to be used in the benchmark.",
        )

        spec.outline(
            cls.setup,
            cls.submit_calculations,
            cls.finalize,
        )
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
        self.ctx.max_tasks = self.inputs.code.computer.get_default_mpiprocs_per_machine()

        self.ctx.files = {
            "basis": orm.SinglefileData(
                file=pathlib.Path(__file__).parent / "data" / "BASIS_MOLOPT"
            ),
            "pseudo": orm.SinglefileData(
                file=pathlib.Path(__file__).parent / "data" / "POTENTIAL"
            ),
            "mpswrapper": orm.SinglefileData(
                file=pathlib.Path(__file__).parent / "data" / "mps-wrapper.sh"
            ),
        }
        self.ctx.input_dict = cp2k_utils.load_protocol(
                "benchmarks.yml", self.inputs.protocol.value)
        # UKS.
        magnetization_per_site = [0 for i in range(len(self.inputs.structure.sites))]
        multiplicity = getattr(self.inputs, "multiplicity", None)
        if multiplicity:
            #magnetization_per_site = self.ctx.dft_params["magnetization_per_site"]
            self.ctx.input_dict["FORCE_EVAL"]["DFT"]["UKS"] = ".TRUE."
            self.ctx.input_dict["FORCE_EVAL"]["DFT"]["MULTIPLICITY"] = multiplicity

        # Get initial magnetization.
        structure_with_tags, kinds_dict = cp2k_utils.determine_kinds(
            self.inputs.structure, magnetization_per_site
        )

        ase_atoms = structure_with_tags.get_ase()

        self.ctx.structure_with_tags = ase_atoms
        self.ctx.kinds_section = cp2k_utils.get_kinds_section(
            kinds_dict, protocol="gpw"
        )
        cp2k_utils.dict_merge(self.ctx.input_dict, self.ctx.kinds_section)

        # Overwrite cutoff if given in dft_params.
        cutoff = getattr(self.inputs, "cutoff", cp2k_utils.get_cutoff(structure=self.inputs.structure))

        self.ctx.input_dict["FORCE_EVAL"]["DFT"]["MGRID"]["CUTOFF"] = cutoff


        return engine.ExitCode(0)


    def submit_calculations(self):
        input_dict = self.ctx.input_dict

        for nnodes in self.inputs.list_nodes:
            if nnodes <= 8:
                mywall=50
            else:
	            mywall=20
             
            # Loop for mpi tasks 
            for ntasks in self.inputs.list_tasks_per_node:
                for nthreads in self.inputs.list_threads_per_task:
                    # Loop for threads,check that nthreads * ntasks <= max_tasks
                    if  nthreads<=self.ctx.max_tasks/ntasks :
                        # Prepare the builder.
                        builder = Cp2kCalculation.get_builder()
                        builder.code = self.inputs.code
                        builder.structure = self.inputs.structure
                        builder.file = self.ctx.files

                        # Options.
                        builder.metadata.options =  {
                        "max_wallclock_seconds": self.inputs.wallclock.value,
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
                            **{
                                f"run_{nnodes}_{ntasks}_{nthreads}": 
                                    submitted_calculation
                                
                            }
                        )


    def finalize(self):
        self.report("Finalizing...")
        result = orm.Dict(dict={})
        
        for nnodes in self.inputs.list_nodes:
            # Loop for mpi tasks 
            for ntasks in self.inputs.list_tasks_per_node:
                # Loop for mpi tasks
                for nthreads in self.inputs.list_threads_per_task:
                    # Check that nthreads * ntasks <= 72
                    if  nthreads<=self.ctx.max_tasks/ntasks : 
                        current_calc = getattr(self.ctx, f"run_{nnodes}_{ntasks}_{nthreads}")                               
                        if not current_calc.is_finished_ok:
                            self.report(f"One of the calculations failed: run_{nnodes}_{ntasks}_{nthreads}.")
                            folder_data = None
                        else:
                            folder_data = current_calc.outputs.retrieved
                        
                        result[f"{nnodes}_{ntasks}_{nthreads}"] = get_timing_from_FolderData(folder_data).value
        result.store()                
        self.out('timings', result)
        self.out('report',analyze_speedup(result))
        return engine.ExitCode(0)
