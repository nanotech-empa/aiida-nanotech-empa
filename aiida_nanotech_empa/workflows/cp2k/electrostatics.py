"""Explicit CP2K electrostatics and geometry checks, without implicit resizing.

The preparation rule retains the standard molecular padding of 15 Å. It is a
geometrical precaution, not a measurement of the electronic-density extent;
vacuum and electronic-density convergence remain the caller's responsibility.
"""

from copy import deepcopy

import numpy as np

PERIODICITIES = ("NONE", "X", "Y", "Z", "XY", "XZ", "YZ", "XYZ")
SOLVER_DIMENSIONS = {
    "PERIODIC": (3,),
    "MT": (0, 2),
    "ANALYTIC": (0, 1, 2),
    "MULTIPOLE": (0,),
    "WAVELET": (0, 2, 3),
    "IMPLICIT": (0, 1, 2, 3),
}
MT_PADDING_ANGSTROM = 15.0


def poisson_settings(dft_params):
    """Return an explicit POISSON section, rejecting inconsistent overrides."""
    periodic = str(dft_params.get("periodic", "XYZ")).upper()
    if periodic not in PERIODICITIES:
        raise ValueError(f"Unknown CP2K periodicity: {periodic}")
    dimension = 0 if periodic == "NONE" else len(periodic)
    solver = str(
        dft_params.get("poisson_solver", "PERIODIC" if dimension == 3 else "MT")
    ).upper()
    if solver not in SOLVER_DIMENSIONS:
        raise ValueError(f"Unknown CP2K Poisson solver: {solver}")
    if dimension not in SOLVER_DIMENSIONS[solver]:
        raise ValueError(
            f"CP2K {solver} does not support {dimension}D periodicity ({periodic}). "
            "MT is implemented only for 0D and 2D; use an explicit supported solver for 1D."
        )
    extra = deepcopy(dft_params.get("poisson_settings", {}))
    if not isinstance(extra, dict):
        raise ValueError(
            "poisson_settings must be a dictionary of CP2K POISSON settings"
        )
    reserved = {"PERIODIC", "POISSON_SOLVER", "POISSON", "PSOLVER"}
    if any(not isinstance(key, str) or key.upper() in reserved for key in extra):
        raise ValueError(
            "Set periodic and poisson_solver explicitly, outside poisson_settings"
        )
    return {**extra, "PERIODIC": periodic, "POISSON_SOLVER": solver}


def cell_geometry(atoms):
    """Return face separations and atomic spans in the reciprocal face normals."""
    cell = np.asarray(atoms.cell, dtype=float)
    if not len(atoms) or not np.isfinite(atoms.positions).all():
        raise ValueError("Electrostatics requires nonempty, finite atomic coordinates")
    if not np.isfinite(cell).all() or abs(np.linalg.det(cell)) < 1e-10:
        raise ValueError("Electrostatics requires a finite, nonzero-volume cell")
    reciprocal = np.linalg.inv(cell)
    lengths = np.linalg.norm(reciprocal, axis=0)
    heights = 1.0 / lengths
    normals = reciprocal / lengths
    projected = atoms.positions @ normals
    return heights, np.ptp(projected, axis=0), projected


def isolated_axes(periodic):
    return [i for i, axis in enumerate("XYZ") if axis not in periodic]


def prepare_cell(atoms, settings, padding_angstrom=MT_PADDING_ANGSTROM):
    """Prepare a copy only when requested by the caller; preserve periodic vectors.

    Cell angles and interatomic displacements are retained. Only isolated cell
    vectors may grow, and centering is a rigid translation in those directions.
    """
    settings = poisson_settings(settings)
    if not np.isfinite(padding_angstrom) or padding_angstrom <= 0:
        raise ValueError(
            "The explicit density-tail padding must be positive and finite"
        )
    result = atoms.copy()
    axes = isolated_axes(settings["PERIODIC"])
    factor = 2.0 if settings["POISSON_SOLVER"] == "MT" else 1.0
    if settings["PERIODIC"] == "NONE" and abs(np.linalg.det(result.cell)) < 1e-10:
        if not len(result) or not np.isfinite(result.positions).all():
            raise ValueError(
                "Electrostatics requires nonempty, finite atomic coordinates"
            )
        result.cell = factor * np.ptp(result.positions, axis=0) + padding_angstrom
    heights, spans, _ = cell_geometry(result)
    scale = np.ones(3)
    for i in axes:
        scale[i] = max(1.0, (factor * spans[i] + padding_angstrom) / heights[i])
    result.set_cell(np.asarray(result.cell) * scale[:, None], scale_atoms=False)
    fractional = result.positions @ np.linalg.inv(result.cell)
    translation = np.zeros(3)
    for i in axes:
        translation[i] = 0.5 - (fractional[:, i].min() + fractional[:, i].max()) / 2
    result.positions += translation @ np.asarray(result.cell)
    result.pbc = [axis in settings["PERIODIC"] for axis in "XYZ"]
    return result


def prepare_cells(geometries, settings, padding_angstrom=MT_PADDING_ANGSTROM):
    """Explicitly prepare multiple geometries with one cell and rigid translation."""
    if not geometries:
        raise ValueError("Supply at least one geometry")
    if any(not np.allclose(g.cell, geometries[0].cell) for g in geometries):
        raise ValueError("All input replicas must use the same original cell")
    union = geometries[0].copy()
    for geometry in geometries[1:]:
        union += geometry
    prepared = prepare_cell(union, settings, padding_angstrom=padding_angstrom)
    translation = prepared.positions[0] - union.positions[0]
    result = []
    for geometry in geometries:
        atoms = geometry.copy()
        atoms.cell = prepared.cell
        atoms.positions += translation
        atoms.pbc = prepared.pbc
        validate_cell(atoms, settings)
        result.append(atoms)
    return result


def validate_cell(atoms, dft_params):
    """Reject unsafe MT cells without changing their geometry or coordinates."""
    settings = poisson_settings(dft_params)
    heights, spans, projected = cell_geometry(atoms)
    if settings["POISSON_SOLVER"] != "MT":
        return
    for i in isolated_axes(settings["PERIODIC"]):
        required = 2 * spans[i]
        if heights[i] <= required + 1e-7:
            raise ValueError(
                f"MT cell too small along {('A', 'B', 'C')[i]}: face separation "
                f"{heights[i]:.3f} Å; MT requires more than "
                f"{required:.3f} Å (twice the atomic span), plus room for density tails. "
                "The GUI prepares twice the atomic span plus 15 Å. "
                "Prepare and review a larger cell explicitly; the workflow will not resize it."
            )
        if projected[:, i].min() < -1e-7 or projected[:, i].max() > heights[i] + 1e-7:
            raise ValueError(
                "MT atoms lie outside an isolated cell face; center the structure explicitly"
            )


def apply_poisson(parameters, atoms, dft_params):
    """Set both CP2K periodicities, preserving all supplied atomic coordinates."""
    settings = poisson_settings(dft_params)
    validate_cell(atoms, dft_params)
    force_eval = parameters["FORCE_EVAL"]
    force_eval["SUBSYS"]["CELL"]["PERIODIC"] = settings["PERIODIC"]
    force_eval["DFT"]["POISSON"] = settings
