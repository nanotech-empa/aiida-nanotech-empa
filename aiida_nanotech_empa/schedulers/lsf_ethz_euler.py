from aiida.plugins import SchedulerFactory

LsfScheduler = SchedulerFactory("lsf")


class ETHZEulerLsfScheduler(LsfScheduler):
    """
    The ETHZ Euler LSF scheduler requires memory and scratch space to be
    reserved with the line 
    #BSUB -R "rusage[mem=X,scratch=Y]"
    where X and Y are specified in units of MB per cpu
    """
    def _get_submit_script_header(self, job_tmpl):
        lsf_script_lines = super()._get_submit_script_header(
            job_tmpl).splitlines()

        physical_memory_kb = int(job_tmpl.max_memory_kb)
        num_mpiprocs = job_tmpl.job_resource.get_tot_num_mpiprocs()
        mem_per_proc_mb = physical_memory_kb // (1024 * num_mpiprocs)

        new_lines = []
        for line in lsf_script_lines:
            if line.startswith("#BSUB -M"):
                # Skip the BSUB -M line
                continue
            if not line.strip():
                # Add the rusage line before the first empty line
                new_lines.append(
                    '#BSUB -R \"rusage[mem={},scratch={}]\"'.format(
                        mem_per_proc_mb, 2 * mem_per_proc_mb))
            new_lines.append(line)

        return "\n".join(lsf_script_lines)
