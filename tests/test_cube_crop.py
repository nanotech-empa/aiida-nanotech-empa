"""Keep cropped density samples and translated atoms on the same physical grid."""

import copy
import io

import ase
import numpy as np
import pytest
from aiida_gaussian.utils.cube import Cube

from aiida_nanotech_empa.helpers import ANG_TO_BOHR
from aiida_nanotech_empa.parsers.pp_parser import PpParser
from aiida_nanotech_empa.utils.cube_utils import crop_cube


def qe_cube():
    """QE's six-decimal voxel spacing reproduces the normal ribbon failure."""
    shape = (54, 192, 192)
    voxel = np.diag([0.149078, 0.147635, 0.147635])
    positions_bohr = np.array(
        [[1.341706, 17.658799, 14.172946], [1.029901, 8.903252, 14.172946]]
    )
    return Cube(
        title="Grid registration regression",
        comment="Analytic voxel values on a QE-format grid",
        ase_atoms=ase.Atoms("CH", positions=positions_bohr / ANG_TO_BOHR),
        origin=np.zeros(3),
        cell=voxel * np.array(shape)[:, None],
        data=np.arange(np.prod(shape), dtype=float).reshape(shape),
    )


def test_qe_crop_preserves_samples_and_atom_registration():
    cube = qe_cube()
    original = copy.deepcopy(cube)
    crop_cube(cube, y_crop=1.8, z_crop=3.1)

    # The physical bounds snap to these integer indices. In particular, the
    # lower Z bound recovers as 55.99999999999999 before integer conversion.
    start = np.array([0, 37, 56])
    end = np.array([54, 143, 136])
    np.testing.assert_array_equal(cube.data, original.data[:, 37:143, 56:136])
    voxel_bohr = original.cell / np.array(original.data.shape)[:, None]
    translation = start @ voxel_bohr / ANG_TO_BOHR
    np.testing.assert_allclose(
        cube.ase_atoms.positions + translation,
        original.ase_atoms.positions,
        rtol=0,
        atol=2e-15,
    )
    np.testing.assert_allclose(cube.cell, voxel_bohr * (end - start)[:, None])


@pytest.mark.parametrize(
    ("margins", "start", "end"),
    [
        ((None, None, None), (0, 0, 0), (14, 15, 16)),
        ((0, 0, 0), (0, 0, 0), (14, 15, 16)),
        ((0.2, None, None), (2, 0, 0), (11, 15, 16)),
        ((None, 0.2, None), (0, 3, 0), (14, 12, 16)),
        ((None, None, 0.2), (0, 0, 4), (14, 15, 13)),
        (((0.1, 0.3), (0.2, 0.1), (0.3, 0.2)), (3, 3, 3), (12, 11, 13)),
        ((100, 100, 100), (0, 0, 0), (14, 15, 16)),
    ],
)
@pytest.mark.parametrize("origin", [(0.0, 0.0, 0.0), (0.3, 0.5, 0.7)])
def test_crop_bounds_and_coordinates(margins, start, end, origin):
    shape = np.array([14, 15, 16])
    spacing_angstrom = 0.1
    origin = np.array(origin)
    positions = origin + np.array([[4, 5, 6], [9, 10, 11]]) * spacing_angstrom
    data = np.arange(np.prod(shape), dtype=float).reshape(shape)
    cube = Cube(
        ase_atoms=ase.Atoms("CH", positions=positions),
        origin=origin * ANG_TO_BOHR,
        cell=np.diag(shape * spacing_angstrom) * ANG_TO_BOHR,
        data=data.copy(),
    )

    crop_cube(cube, *margins)

    expected = data[tuple(slice(a, b) for a, b in zip(start, end))]
    np.testing.assert_array_equal(cube.data, expected)
    translation = origin + np.array(start) * spacing_angstrom
    np.testing.assert_allclose(cube.ase_atoms.positions + translation, positions)
    np.testing.assert_allclose(
        cube.cell / ANG_TO_BOHR, np.diag(np.array(end) - start) * spacing_angstrom
    )


def test_pp_parser_keeps_cropped_grid_and_atomic_coordinates_together(aiida_profile):
    cube = qe_cube()
    original = copy.deepcopy(cube)
    stream = io.StringIO()
    stream.write("Grid registration regression\nAnalytic QE-format density\n")
    stream.write(f"{len(cube.ase_atoms)} 0 0 0\n")
    voxel = cube.cell / np.array(cube.data.shape)[:, None]
    for n, vector in zip(cube.data.shape, voxel):
        stream.write(f"{n} {vector[0]:.6f} {vector[1]:.6f} {vector[2]:.6f}\n")
    for atom in cube.ase_atoms:
        xyz = atom.position * ANG_TO_BOHR
        stream.write(f"{atom.number} 0 {xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}\n")
    # Unique integer voxel values make even a one-voxel shift observable.
    stream.write(" ".join(map(str, cube.data.ravel())))

    parsed = PpParser.parse_gaussian(stream.getvalue(), "e/bohr^3")

    np.testing.assert_array_equal(
        parsed.get_array("data"), original.data[:, 37:143, 56:136]
    )
    np.testing.assert_array_equal(parsed.get_array("atomic_numbers"), [6, 1])
    translation_bohr = np.array([0, 37, 56]) @ voxel
    np.testing.assert_allclose(
        parsed.get_array("coordinates") + translation_bohr,
        original.ase_atoms.positions * ANG_TO_BOHR,
        rtol=0,
        atol=2e-14,
    )
    np.testing.assert_allclose(parsed.get_array("voxel"), voxel)
    assert parsed.get_array("data_units") == "e/bohr^3"
    assert parsed.get_array("coordinates_units") == "bohr"
