import numpy as np

# -------------------------------------------------------------------------------------
# Memory determination
# -------------------------------------------------------------------------------------
# The amount of memory needs to be specified in the Gaussian input script, otherwise
# only a default small amount is used, exact value depends on the Gaussian version.
#
# The amount of memory specified to the job scheduler needs to be higher than the amount
# specified to Gaussian. A good rule of thumb is that the scheduler should have 25% more
# memory. However, the direct scheduler needs special treatment as long as the following
# issue is not solved: https://github.com/aiidateam/aiida-core/issues/4526


def _get_total_mem_kb(gaussian_mem_mb, computer):
    if computer.scheduler_type == 'direct':
        # For direct scheduler, should ask for extra ~1.5GB
        return (gaussian_mem_mb + 1536) * 1024
    return int(1.25 * gaussian_mem_mb + 100) * 1024


def _get_gaussian_mem_mb(total_mem_kb, computer):
    if computer.scheduler_type == 'direct':
        return total_mem_kb // 1024 - 1536
    return (total_mem_kb // 1024 - 100) // 1.25


# -------------------------------------------------------------------------------------


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

    self_.ctx.comp = self_.inputs.gaussian_code.computer


def determine_comp_resources(num_atoms, basis_set=""):

    if basis_set.lower() in ("sto-3g", "sv", "svp", "def2sv", "def2svp"):
        num_cores = int(np.round(num_atoms / 20))
        mem_per_core = 512
    else:
        num_cores = int(np.round(num_atoms / 16))
        mem_per_core = 4096

    num_cores = max(1, num_cores)
    num_cores = min(24, num_cores)

    memory_mb = num_cores * mem_per_core

    return num_cores, memory_mb


def get_default_metadata_options(num_atoms, computer, basis_set):

    num_cores, memory_mb = determine_comp_resources(num_atoms, basis_set)

    options = {}

    options['max_memory_kb'] = _get_total_mem_kb(memory_mb, computer)

    options['max_wallclock_seconds'] = 24 * 60 * 60

    if 'lsf' in computer.scheduler_type:
        # LSF scheduler has some peculiarities:
        # 'num_machines' can not be set
        # tot_num_mpiprocs determines the number of cores
        options['resources'] = {
            "tot_num_mpiprocs": num_cores,
        }
    else:
        options['resources'] = {
            'num_machines': 1,
            "tot_num_mpiprocs": 1,
            'num_cores_per_machine': num_cores
        }

    return options


def validate_metadata_options(options, computer):

    if 'resources' not in options:
        return "'resources' needs to be set"

    res = options['resources']

    if get_total_number_of_cores(res, computer) is None:
        return 'num_cores can not be determined from the inputted resources'

    if 'max_memory_kb' in options:
        # see if at least 200 MB memory will be accessible to Gaussian
        if _get_gaussian_mem_mb(options['max_memory_kb'], computer) < 200:
            return "Too little memory specified."

    if 'num_machines' in res and res['num_machines'] != 1:
        # if 'num_machines' is set, it should be 1 as Gaussian doesn't support MPI
        return "'num_machines' either needs to be unset or set to 1"

    return None


def determine_metadata_options(self_):
    """ returns False if something failed """

    if 'options' in self_.inputs:
        val = validate_metadata_options(dict(self_.inputs.options),
                                        self_.ctx.comp)
        if val is not None:
            self_.report("Error: " + val)
            return False
        self_.ctx.metadata_options = dict(self_.inputs.options)

    else:
        bset = ""
        if 'basis_set' in self_.ctx:
            bset = self_.ctx.basis_set
        elif 'basis_set_scf' in self_.inputs:
            bset = self_.inputs.basis_set_scf.value
        elif 'basis_set' in self_.inputs:
            bset = self_.inputs.basis_set.value

        self_.ctx.metadata_options = get_default_metadata_options(
            self_.ctx.n_atoms, self_.ctx.comp, bset)

    # Always use the gaussian_advanced_parser
    self_.ctx.metadata_options['parser_name'] = 'gaussian.advanced'

    return True


def get_total_number_of_cores(resources, computer):

    num_machines = 1
    if 'num_machines' in resources:
        num_machines = resources['num_machines']

    if 'num_cores_per_machine' in resources:
        return num_machines * resources['num_cores_per_machine']

    tot_num_mpiprocs = 1
    if 'tot_num_mpiprocs' in resources:
        tot_num_mpiprocs = resources['tot_num_mpiprocs']
    elif 'num_mpiprocs_per_machine' in resources:
        tot_num_mpiprocs = resources['num_mpiprocs_per_machine'] * num_machines
    else:
        def_mppm = computer.get_default_mpiprocs_per_machine()
        if def_mppm is not None:
            tot_num_mpiprocs = def_mppm * num_machines

    if 'num_cores_per_mpiproc' in resources:
        return tot_num_mpiprocs * resources['num_cores_per_mpiproc']

    return tot_num_mpiprocs


def get_gaussian_cores_and_memory(options, computer):

    num_cores = get_total_number_of_cores(options['resources'], computer)

    if 'max_memory_kb' not in options:
        # If no memory is specified for the scheduler, set 2GB memory to Gaussian
        memory_mb = 2048
    else:
        memory_mb = _get_gaussian_mem_mb(options['max_memory_kb'], computer)

    return num_cores, memory_mb
