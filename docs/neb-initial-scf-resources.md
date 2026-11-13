# Initial SCF resources for NEB

When no compatible wavefunction is available, `Cp2kNebWorkChain` first runs a
single SCF calculation on the initial geometry. That calculation uses exactly
`NPROC_REP` MPI ranks, the allocation of one replica environment. The later BAND
calculation retains its complete requested allocation.

The initial SCF uses the fewest whole nodes that fit those ranks without exceeding
the BAND node count or MPI tasks per node. The tasks per node must divide
`NPROC_REP` exactly. Threads per MPI task, walltime, memory and scientific inputs
are retained. AiiDA's scheduler resource validator checks the resulting layout.
Inconsistent or impossible allocations return exit code 392 with a diagnostic
before submitting a child. A compatible reused WFN skips this SCF stage.

| BAND allocation | Replica environments | NPROC_REP | Initial SCF |
| --- | --- | --- | --- |
| 1 node × 20 MPI | 5 | 4 | 1 node × 4 MPI |
| 2 nodes × 4 MPI | 2 | 4 | 1 node × 4 MPI |
| 2 nodes × 4 MPI | 1 | 8 | 2 nodes × 4 MPI |
| 4 nodes × 6 MPI | 3 | 8 | 2 nodes × 4 MPI |

## Small localhost GUI regression

Use the normal Surfaces NEB form, a bare methane structure built from SMILES `C`
and a 12 Å cubic cell. Store it without first running a calculation. Create a
second endpoint by moving one H atom by 0.03 Å along z in the normal editor and
storing the edit. Both sources must lack a compatible calculation/WFN ancestor.
Keep a separate unfinished scientific draft open rather than overwriting it.

1. In a fresh NEB form, load the saved endpoints by their database identifiers.
   Leave the restart field blank. Insert an intermediate after the initial row
   and interpolate at local factor 1/3. Insert a second row after it and use local
   factor 0.5 between that row and the final endpoint. This gives four explicit
   images at fractions 0, 1/3, 2/3 and 1. Show each row and verify atom order.
   An alternative is two explicit endpoints with the count raised to four; CP2K
   then bisects the widest remaining gaps, so fractions need not be uniform.
2. Choose Standard PBE with dispersion, restricted charge zero and the explicit
   original 12 Å XYZ/PERIODIC calculation model. Keep Align Frames, Rotate Frames
   and Optimize Endpoints off. Use CI-NEB, spring 0.05 and five steps before CI.
3. Choose localhost, one node, four MPI tasks, one thread per task and 600 seconds.
   Set two replicas per group. After Show/tab changes, check four replicas,
   two groups and `NPROC_REP=2` agree in the control and resource helper.
4. Review and submit once through the GUI. Expect strictly serial initial SCF
   (one node × two MPI × one thread), followed only after successful SCF by BAND
   (one node × four MPI × one thread). No compatible WFN is fabricated or removed.
5. Inspect native resources, convergence, generated WFN copy commands and four
   replica results. Record warnings or partial results as observed. This is an
   allocation/functionality regression, not a methane reaction-barrier validation.
   Each native stage has its requested 600-second scheduler limit; the workflow
   does not introduce a retry loop. A failed initial SCF must prevent BAND.
6. Record the source fingerprint, result UUID, exported evidence and reopening
   route. Close completed-test tabs while keeping active drafts and unrelated tabs.

The maintained automated test intercepts actual child builders on isolated AiiDA
storage. It exercises both WFN branches, the fifteen GUI resource layouts,
additional integral/fractional multi-node cases and invalid inputs. It never runs
CP2K. Actual GUI/native evidence remains a separate qualification step.

References: [CP2K BAND input](https://manual.cp2k.org/cp2k-2023_1-branch/CP2K_INPUT/MOTION/BAND.html)
and [current interpolation description](https://manual.cp2k.org/trunk/CP2K_INPUT/MOTION/BAND.html).
