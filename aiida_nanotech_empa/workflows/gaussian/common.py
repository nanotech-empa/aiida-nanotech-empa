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


def setup_context_variables(self_):

    pymatgen_structure = self_.inputs.structure.get_pymatgen_molecule()
    self_.ctx.n_atoms = pymatgen_structure.num_sites
    self_.ctx.n_electrons = pymatgen_structure.nelectrons

    if self_.inputs.multiplicity.value == 0:
        # RKS calculation
        self_.ctx.functional = self_.inputs.functional.value
        self_.ctx.mult = 1
    else:
        # UKS calculation
        self_.ctx.functional = 'u' + self_.inputs.functional.value
        self_.ctx.mult = self_.inputs.multiplicity.value

    bset = ""
    if "basis_set" in self_.inputs:
        bset = self_.inputs.basis_set.value
    elif "basis_set_scf" in self_.inputs:
        bset = self_.inputs.basis_set_scf.value

    num_cores, memory_mb = determine_comp_resources(self_.ctx.n_atoms, bset)

    self_.ctx.num_cores = num_cores
    self_.ctx.memory_mb = memory_mb

    self_.ctx.link0 = {
        '%chk': 'aiida.chk',
        '%mem': "%dMB" % memory_mb,
        '%nprocshared': str(num_cores),
    }

    self_.ctx.comp = self_.inputs.gaussian_code.computer


def set_metadata(builder_metadata, num_atoms, computer):

    num_cores, memory_mb = determine_comp_resources(num_atoms)

    if computer.scheduler_type == 'direct':
        # For direct scheduler, should ask for extra ~1.5GB for libraries etc
        builder_metadata.options.max_memory_kb = (memory_mb + 1536) * 1024
    else:
        builder_metadata.options.max_memory_kb = (memory_mb + 512) * 1024

    builder_metadata.options.max_wallclock_seconds = 24 * 60 * 60

    builder_metadata.options.resources = {
        "tot_num_mpiprocs": num_cores,
    }

    if computer.scheduler_type != 'lsf':
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
