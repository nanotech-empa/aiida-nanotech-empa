from aiida.schedulers.plugins.slurm import SlurmScheduler


class ETHZEulerSlurmScheduler(SlurmScheduler):
    """The ETHZ Euler SLURM scheduler requires -N for the number of nodes and -n for th enumber of cores per node."""

    def _get_submit_script_header(self, job_tmpl):
        if job_tmpl.max_memory_kb:
            slurm_script_lines = (
                super()._get_submit_script_header(job_tmpl).splitlines()
            )
            job_tmpl.job_resource.get_tot_num_mpiprocs()
            new_lines = []
            for line in slurm_script_lines:
                if line.startswith("#SBATCH --nodes="):
                    # Change nodes into N.
                    nodes = int(line.split("=")[-1])
                    new_lines.append(f"#SBATCH -N  + {nodes}")
                elif line.startswith("#SBATCH --ntasks-per-node="):
                    # Change ntasks-per-node into n.
                    task_per_node = int(line.split("=")[-1])
                    new_lines.append(f"#SBATCH -n {task_per_node}")
                elif line.startswith("#SBATCH --mem"):
                    # Change memory into mem-per-cpu.
                    mem = int(line.split("=")[-1])
                    new_lines.append(
                        "#SBATCH --mem-per-cpu="
                        + str(int(mem / (task_per_node * nodes)))
                    )
                else:
                    new_lines.append(line)

            return "\n".join(new_lines)

        # If memory is not specified, just use the default LSF script.
        return super()._get_submit_script_header(job_tmpl)
