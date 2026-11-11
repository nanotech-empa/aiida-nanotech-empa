"""Actual adsorption outline, linked model controls and builders; no native jobs."""

import copy
import io
from types import SimpleNamespace as NS

import numpy as np
import pytest
import test_fragment_serial
from aiida import engine, orm, plugins
from aiida_cp2k.parsers import Cp2kBaseParser
from test_fragment_serial import EVENTS, TestWorkflow, run_case

from aiida_nanotech_empa.utils.cp2k_results import calculation_boundary, periodic_pbc
from aiida_nanotech_empa.workflows.cp2k.electrostatics import (
    PERIODICITIES,
    SOLVER_DIMENSIONS,
)
from aiida_nanotech_empa.workflows.cp2k.fragment_separation import (
    Cp2kFragmentSeparationWorkChain,
)

source_inputs = test_fragment_serial.source_inputs

pytest_plugins = ["aiida.tools.pytest_fixtures"]


def solver_for(axes):
    return "PERIODIC" if axes == "XYZ" else "ANALYTIC" if len(axes) == 1 else "MT"


@pytest.fixture
def model_inputs(source_inputs):
    inputs = dict(source_inputs)
    atoms = inputs["structure"].get_ase()
    atoms.positions[3:, 0] -= (
        3  # Compact synthetic dimer fits MT in an explicit 12 Å cell.
    )
    atoms.pbc = (
        False  # Deliberately different source metadata from some effective models.
    )
    inputs["structure"] = orm.StructureData(ase=atoms).store()
    return inputs


@pytest.mark.parametrize("protocol", ["standard", "low_accuracy"])
@pytest.mark.parametrize("axes", PERIODICITIES)
def test_real_components_working_and_parsed_result_model(model_inputs, axes, protocol):
    inputs = dict(model_inputs)
    inputs["protocol"] = orm.Str(protocol)
    dft = inputs["dft_params"].get_dict()
    dft.update(periodic=axes, poisson_solver=solver_for(axes))
    inputs["dft_params"] = orm.Dict(dft)
    source_before = copy.deepcopy(inputs["structure"].base.attributes.all)
    _, node, _, payloads, peak = run_case(inputs, True)
    assert node.is_finished_ok and peak == 1
    for row in payloads[:-1]:
        model = calculation_boundary(row["parameters"])
        assert model["periodic"] == axes and model["poisson_solver"] == solver_for(axes)
        assert tuple(row["pbc"]) == periodic_pbc(axes)
        np.testing.assert_array_equal(row["cell"], inputs["structure"].cell)
        indices = (
            slice(None)
            if row["name"].endswith("all")
            else slice(0, 3)
            if row["name"].endswith("water_a")
            else slice(3, 6)
        )
        np.testing.assert_array_equal(
            row["positions"], inputs["structure"].get_ase().positions[indices]
        )
        # Exercise the installed final-structure parser on a synthetic restart
        # with this exact native CELL model. This does not claim native execution.
        text = (
            "\n&CELL\n A 12 0 0\n B 0 12 0\n C 0 0 12\n PERIODIC "
            + axes
            + "\n&END CELL\n&COORD\n H 6 6 6\n&END COORD\n"
        )
        retrieved = orm.FolderData()
        retrieved.base.repository.put_object_from_filelike(
            io.StringIO(text), "aiida-1.restart"
        )
        parser_context = NS(
            node=NS(process_class=plugins.CalculationFactory("cp2k")),
            retrieved=retrieved,
        )
        result = Cp2kBaseParser._parse_final_structure(parser_context)
        assert result.pbc == periodic_pbc(axes)
    assert inputs["structure"].base.attributes.all == source_before


@pytest.mark.parametrize("axes", PERIODICITIES)
@pytest.mark.parametrize("solver", SOLVER_DIMENSIONS)
def test_preflight_solver_axis_matrix(model_inputs, axes, solver):
    inputs = dict(model_inputs)
    dft = inputs["dft_params"].get_dict()
    dft.update(periodic=axes, poisson_solver=solver)
    inputs["dft_params"] = orm.Dict(dft)
    valid = sum(periodic_pbc(axes)) in SOLVER_DIMENSIONS[solver] and not (
        solver == "WAVELET" and len(axes) == 2 and axes != "XZ"
    )
    error = Cp2kFragmentSeparationWorkChain.validate_inputs(inputs)
    if valid:
        assert error is None
    else:
        assert error and ("support" in error or "implemented" in error)


@pytest.mark.parametrize("kind", ["solver", "cell", "poisson", "fft", "small_cell"])
def test_reject_invalid_model_before_any_child(model_inputs, kind):
    inputs = dict(model_inputs)
    dft = inputs["dft_params"].get_dict()
    if kind == "solver":
        dft.update(periodic="Y", poisson_solver="MT")
    elif kind == "small_cell":
        dft.update(periodic="NONE", poisson_solver="MT")
        atoms = inputs["structure"].get_ase()
        atoms.cell = [2, 2, 2]
        inputs["structure"] = orm.StructureData(ase=atoms)
    else:
        if kind == "cell":
            override = {"FORCE_EVAL": {"SUBSYS": {"CELL": {"PERIODIC": "NONE"}}}}
        elif kind == "poisson":
            override = {
                "FORCE_EVAL": {
                    "SUBSYS": {"CELL": {"PERIODIC": "NONE"}},
                    "DFT": {"POISSON": {"PERIODIC": "NONE", "POISSON_SOLVER": "MT"}},
                }
            }
        else:
            dft.update(periodic="XZ", poisson_solver="WAVELET")
            override = {"GLOBAL": {"PREFERRED_FFT_LIBRARY": "FFTW3"}}
        inputs["auxilary_dictionaries"] = {"override": orm.Dict(override)}
    inputs["dft_params"] = orm.Dict(dft)
    EVENTS.clear()
    with pytest.raises(ValueError):
        engine.run_get_node(TestWorkflow, **inputs)
    assert EVENTS == []


def test_compatible_auxiliary_settings_and_omitted_model(model_inputs):
    inputs = dict(model_inputs)
    dft = inputs["dft_params"].get_dict()
    dft.pop("periodic")
    inputs["dft_params"] = orm.Dict(dft)
    inputs["auxilary_dictionaries"] = {
        "extra": orm.Dict(
            {
                "FORCE_EVAL": {
                    "DFT": {
                        "POISSON": {"PERIODIC": "XYZ", "POISSON_SOLVER": "PERIODIC"}
                    }
                },
                "MOTION": {"GEO_OPT": {"MAX_ITER": 12}},
            }
        )
    }
    _, node, _, payloads, _ = run_case(inputs, True)
    assert node.is_finished_ok
    for row in payloads[:-1]:
        assert tuple(row["pbc"]) == (True, True, True)
        assert row["parameters"]["MOTION"]["GEO_OPT"]["MAX_ITER"] == 12
