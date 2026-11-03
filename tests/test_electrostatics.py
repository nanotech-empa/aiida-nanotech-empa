"""Physical cell geometry, supported solvers and preservation at input staging."""

from copy import deepcopy

import numpy as np
import pytest
from aiida import orm
from ase import Atoms

from aiida_nanotech_empa.workflows.cp2k import cp2k_utils
from aiida_nanotech_empa.workflows.cp2k.electrostatics import (
    apply_poisson,
    cell_geometry,
    poisson_settings,
    prepare_cell,
    validate_cell,
)


def molecule():
    return Atoms(
        "H2",
        positions=[[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]],
        cell=[8.0, 9.0, 10.0],
        pbc=True,
    )


@pytest.mark.parametrize("periodic", ["X", "Y", "Z", "XYZ"])
def test_mt_rejects_unsupported_dimension(periodic):
    with pytest.raises(ValueError, match="MT is implemented only for 0D and 2D"):
        poisson_settings({"periodic": periodic, "poisson_solver": "MT"})


def test_preparation_preserves_skew_metric_and_rigid_geometry():
    atoms = molecule()
    atoms.cell = [[8.0, 0.0, 0.0], [3.0, 8.0, 0.0], [1.0, 2.0, 7.0]]
    original = atoms.copy()
    p = {"periodic": "XY", "poisson_solver": "MT"}
    prepared = prepare_cell(atoms, p)
    np.testing.assert_allclose(prepared.cell[:2], atoms.cell[:2])
    np.testing.assert_allclose(prepared.cell.angles(), atoms.cell.angles())
    np.testing.assert_allclose(
        prepared.positions[1] - prepared.positions[0],
        atoms.positions[1] - atoms.positions[0],
    )
    heights, spans, _ = cell_geometry(prepared)
    assert heights[2] == pytest.approx(2 * spans[2] + 15.0)
    validate_cell(prepared, p)
    assert atoms == original


def test_preparation_never_shrinks_a_large_cell():
    atoms = molecule()
    atoms.cell = [60.0, 70.0, 80.0]
    prepared = prepare_cell(atoms, {"periodic": "NONE", "poisson_solver": "MT"})
    np.testing.assert_array_equal(prepared.cell, atoms.cell)


def test_mt_rejects_too_small_cell_without_mutation():
    atoms = molecule()
    atoms.cell = [3.0, 4.0, 4.0]
    before = atoms.copy()
    with pytest.raises(ValueError, match="MT cell too small"):
        validate_cell(atoms, {"periodic": "NONE"})
    assert atoms == before


def test_zero_cell_molecule_can_be_prepared_explicitly():
    atoms = molecule()
    atoms.cell = [0.0, 0.0, 0.0]
    prepared = prepare_cell(atoms, {"periodic": "NONE"})
    validate_cell(prepared, {"periodic": "NONE"})
    assert np.linalg.det(prepared.cell) > 0
    assert np.linalg.det(atoms.cell) == 0


def test_all_trajectory_frames_are_checked(aiida_profile):
    trajectory = orm.TrajectoryData()
    trajectory.set_trajectory(
        symbols=["H", "H"],
        positions=np.array(
            [[[2.0, 2.0, 2.0], [3.0, 2.0, 2.0]], [[0.0, 2.0, 2.0], [7.0, 2.0, 2.0]]]
        ),
        cells=np.array([np.eye(3) * 8, np.eye(3) * 8]),
    )
    message = cp2k_utils.validate_poisson_inputs(
        {
            "trajectory": trajectory,
            "dft_params": orm.Dict(dict={"periodic": "NONE"}),
        }
    )
    assert "MT cell too small" in message


def test_preflight_does_not_submit_or_resize_invalid_input(aiida_profile):
    atoms = molecule()
    atoms.cell = [3.0, 4.0, 4.0]
    source = orm.StructureData(ase=atoms)
    error = cp2k_utils.validate_poisson_inputs(
        {
            "structure": source,
            "dft_params": orm.Dict(dict={"periodic": "NONE"}),
        }
    )
    assert "MT cell too small" in error
    np.testing.assert_array_equal(source.cell, atoms.cell)


@pytest.mark.parametrize(
    "periodic,solver",
    [("NONE", "MT"), ("XY", "MT"), ("Z", "ANALYTIC"), ("XYZ", "PERIODIC")],
)
def test_poisson_settings_are_used_without_moving_atoms(periodic, solver):
    params = {
        "periodic": periodic,
        "poisson_solver": solver,
        "poisson_settings": {"MT": {"ALPHA": 9.0}} if solver == "MT" else {},
    }
    atoms = prepare_cell(molecule(), params)
    before = atoms.copy()
    request = deepcopy(params)
    d = cp2k_utils.load_protocol("geo_opt_protocol.yml", "standard")
    apply_poisson(d, atoms, params)
    assert d["FORCE_EVAL"]["DFT"]["POISSON"] == {
        **params["poisson_settings"],
        "PERIODIC": periodic,
        "POISSON_SOLVER": solver,
    }
    assert d["FORCE_EVAL"]["SUBSYS"]["CELL"]["PERIODIC"] == periodic
    assert atoms == before
    assert params == request


def test_advanced_settings_cannot_override_boundary_conditions():
    for extra in [{"PERIODIC": "X"}, {"psolver": "MT"}, []]:
        with pytest.raises(ValueError):
            poisson_settings({"poisson_settings": extra})


@pytest.mark.parametrize("protocol", ["debug", "standard"])
def test_get_dft_inputs_preserves_explicit_mt_cell(aiida_profile, protocol):
    params = {
        "periodic": "NONE",
        "poisson_solver": "MT",
        "poisson_settings": {"MT": {"ALPHA": 9.0}},
    }
    atoms = prepare_cell(molecule(), params)
    node = orm.StructureData(ase=atoms)
    _, d, staged = cp2k_utils.get_dft_inputs(
        params, node, "geo_opt_protocol.yml", protocol
    )
    np.testing.assert_array_equal(staged.cell, atoms.cell)
    np.testing.assert_array_equal(staged.positions, atoms.positions)
    assert d["FORCE_EVAL"]["DFT"]["POISSON"]["MT"] == {"ALPHA": 9.0}
