from aiida.schedulers.plugins.slurm import SlurmScheduler


class ETHZEulerSlurmScheduler(SlurmScheduler):
    """
    The ETHZ Euler SLURM scheduler requires -N for the number of nodes and -n for th enumber of cores per node 
    """

    def _get_submit_script_header(self, job_tmpl):
        if job_tmpl.max_memory_kb:
            slurm_script_lines = super()._get_submit_script_header(job_tmpl).splitlines()

            physical_memory_kb = int(job_tmpl.max_memory_kb)
            num_mpiprocs = job_tmpl.job_resource.get_tot_num_mpiprocs()
            mem_per_proc_mb = physical_memory_kb // (1024 * num_mpiprocs)

            #rusage_added = False
            new_lines = []
            for line in slurm_script_lines:
                if line.startswith("#SBATCH --nodes="):
                    # change nodes into N
                    N=int(line.split('=')[-1])
                    new_lines.append('#SBATCH -N '+str(N))
                elif line.startswith("#SBATCH --ntasks-per-node="):
                    # change ntasks-per-node into n
                    n=int(line.split('=')[-1])
                    new_lines.append('#SBATCH -n '+str(n)) 
                elif line.startswith("#SBATCH --mem"):
                    # change memory into mem-per-cpu
                    mem=int(line.split('=')[-1])
                    new_lines.append('#SBATCH --mem-per-cpu='+str(int(mem/(n*N))))                   
                else:
                    new_lines.append(line)

            return "\n".join(new_lines)

        # If memory is not specified, just use the default LSF script
        return super()._get_submit_script_header(job_tmpl)
