import numpy as np


def determine_comp_resources(num_atoms, basis_set=""):

    factor = int(np.round(num_atoms / 20))
    factor = max(1, factor)
    factor = min(24, factor)

    num_cores = factor * 1
    if basis_set.lower() in ("sto-3g", "sv", "svp", "def2sv", "def2svp"):
        memory_mb = factor * 512
    else:
        memory_mb = factor * 2048

    return num_cores, memory_mb


def set_metadata(builder_metadata, num_atoms, computer):

    num_cores, memory_mb = determine_comp_resources(num_atoms)

    if computer.get_scheduler_type() == 'direct':
        # For direct scheduler, should ask for extra ~1.5GB for libraries etc
        builder_metadata.options.max_memory_kb = (memory_mb + 1536) * 1024
    else:
        builder_metadata.options.max_memory_kb = (memory_mb + 512) * 1024

    builder_metadata.options.max_wallclock_seconds = 24 * 60 * 60

    builder_metadata.options.resources = {
        "tot_num_mpiprocs": num_cores,
    }

    if computer.get_scheduler_type() != 'lsf':
        # LSF scheduler doesn't work with 'num_machines'
        builder_metadata.options.resources['num_machines'] = 1


def check_if_previous_calc_ok(wc_inst, prev_calc):
    if not prev_calc.is_finished_ok:
        if prev_calc.exit_status is not None and prev_calc.exit_status >= 500:
            wc_inst.report("Warning: previous step: " + prev_calc.exit_message)
        else:
            wc_inst.report("ERROR: previous step: " + prev_calc.exit_message)
            return False
    return True
