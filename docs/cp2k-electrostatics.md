# Explicit CP2K electrostatics and cells

Shared DFT WorkChains no longer resize or centre structures implicitly when
`dft_params.periodic` is `NONE`. The caller must prepare and review its cell
before submission. Existing periodic calculations retain their geometry.

The supported DFT settings are:

```python
params = {
    "periodic": "NONE",  # NONE, X, Y, Z, XY, XZ, YZ or XYZ
    "poisson_solver": "MT",
    "poisson_settings": {},  # Additional CP2K POISSON keywords/subsections
}
```

Both `SUBSYS/CELL/PERIODIC` and `DFT/POISSON/PERIODIC` are set consistently.
Advanced settings cannot override the separately selected periodicity or solver.
Without explicit solver settings, the existing XYZ/PERIODIC and NONE/MT
conventions are retained. A 1D calculation needs an explicit supported solver;
MT is not implemented for 1D. ANALYTIC supports 0D/1D/2D but can converge slowly.
Other solver prerequisites, including CP2K-version/build restrictions, still
apply. See the [CP2K Poisson documentation](https://manual.cp2k.org/trunk/CP2K_INPUT/FORCE_EVAL/DFT/POISSON.html).

For an explicitly requested starting cell:

```python
from aiida_nanotech_empa.workflows.cp2k.electrostatics import prepare_cell

prepared = prepare_cell(atoms, params)
```

This returns a copy. MT preparation uses twice the atomic span plus 15 Å in
isolated directions. Other solvers use the atomic span plus 15 Å. Larger cells
are retained; periodic lattice vectors and cell angles are preserved. The
structure undergoes one rigid translation to centre it in isolated directions.
Face separations and spans are measured in reciprocal face normals, including
for skewed cells. A zero-cell molecule can receive an explicitly created box.

`prepare_cells` applies one common cell and translation to a set of NEB inputs.
The scientific examples now request their debug padding explicitly, rather than
relying on hidden WorkChain changes.

Validation rejects unsupported solver dimensions, invalid cells, an MT face
separation no larger than twice the atomic span, and atoms outside isolated
cell faces. Every supplied trajectory frame and NEB replica is checked. This is
a necessary geometric check, **not proof that the electronic density fits**:
the density extends beyond the atoms, and vacuum/density convergence must still
be checked. The 15 Å preparation padding is a starting rule, not an error bound.

The dedicated GW/image-charge cell construction is a separate method-specific
path; it is not changed by this shared DFT control.
