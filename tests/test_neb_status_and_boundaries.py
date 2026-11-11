"""Invented NEB text and cell fixtures; no native executable or user output."""

import copy
import io
from types import SimpleNamespace

import numpy as np
import pytest
from aiida import orm
from aiida.common import AttributeDict
from aiida.common.links import LinkType
from ase import Atoms

from aiida_nanotech_empa.parsers.cp2k_neb_parser import Cp2kNebParser
from aiida_nanotech_empa.utils.cp2k_results import (
    PERIODICITIES,
    SOLVER_DIMENSIONS,
    calculation_boundary,
    parse_neb_status,
    periodic_pbc,
)
from aiida_nanotech_empa.workflows.cp2k import cp2k_utils
from aiida_nanotech_empa.workflows.cp2k.electrostatics import (
    apply_poisson,
    poisson_settings,
)
from aiida_nanotech_empa.workflows.cp2k.neb_workchain import Cp2kNebWorkChain


def solver_for(periodic):
    return (
        "PERIODIC" if periodic == "XYZ" else "ANALYTIC" if len(periodic) == 1 else "MT"
    )


def atoms():
    return Atoms("H2", positions=[[5, 6, 6], [6, 6, 6]], cell=[12] * 3, pbc=False)


@pytest.mark.parametrize("periodic", PERIODICITIES)
@pytest.mark.parametrize("protocol", ["standard", "low_accuracy"])
def test_working_model_matches_parameters_without_changing_source(
    aiida_profile_clean, periodic, protocol
):
    assert aiida_profile_clean.is_test_profile
    source = orm.StructureData(ase=atoms()).store()
    before = copy.deepcopy(source.base.attributes.all)
    _, params, working = cp2k_utils.get_dft_inputs(
        {"periodic": periodic, "poisson_solver": solver_for(periodic)},
        source,
        "neb_protocol.yml",
        protocol,
    )
    assert tuple(working.pbc) == periodic_pbc(periodic)
    assert calculation_boundary(params)["pbc"] == tuple(working.pbc)
    np.testing.assert_array_equal(working.positions, source.get_ase().positions)
    np.testing.assert_array_equal(working.cell, source.cell)
    assert source.base.attributes.all == before


@pytest.mark.parametrize("periodic", PERIODICITIES)
@pytest.mark.parametrize("solver", SOLVER_DIMENSIONS)
def test_supported_solver_axes(periodic, solver):
    dim = sum(periodic_pbc(periodic))
    valid = dim in SOLVER_DIMENSIONS[solver] and not (
        solver == "WAVELET" and dim == 2 and periodic != "XZ"
    )
    if valid:
        assert (
            poisson_settings({"periodic": periodic, "poisson_solver": solver})[
                "PERIODIC"
            ]
            == periodic
        )
    else:
        with pytest.raises(ValueError):
            poisson_settings({"periodic": periodic, "poisson_solver": solver})


def test_wavelet_fft_and_conflicting_overrides():
    params = {"FORCE_EVAL": {"SUBSYS": {"CELL": {}}, "DFT": {}}}
    apply_poisson(params, atoms(), {"periodic": "XZ", "poisson_solver": "WAVELET"})
    assert params["GLOBAL"]["PREFERRED_FFT_LIBRARY"] == "FFTSG"
    params["GLOBAL"]["PREFERRED_FFT_LIBRARY"] = "FFTW3"
    with pytest.raises(ValueError, match="FFTSG"):
        apply_poisson(params, atoms(), {"periodic": "XZ", "poisson_solver": "WAVELET"})
    for key in ["PERIODIC", "periodic", "PSOLVER", "POISSON", "POISSON_SOLVER"]:
        with pytest.raises(ValueError):
            poisson_settings({"poisson_settings": {key: "NONE"}})


def logs(converged=True, timeout=False):
    flag = "YES" if converged else "NO"
    main = "STEP NUMBER = 0\nDISTANCES REP = 0.1\nENERGIES [au] = -1.0 -0.8 BAND\n"
    main += "\n".join(
        f"{q} = 0.001 [ 0.002 ] ( {flag} )"
        for q in ["RMS DISPLACEMENT", "MAX DISPLACEMENT", "RMS FORCE", "MAX FORCE"]
    )
    return {
        "aiida.out": main,
        "aiida-BAND1.out": "outer SCF loop converged\n",
        "aiida-BAND2.out": "SCF run terminated - exceeded requested execution time\n"
        if timeout
        else "outer SCF loop converged\n",
    }


def test_timeout_electronic_and_band_status_are_independent():
    status = parse_neb_status(logs(converged=False, timeout=True), 2)
    assert status["exceeded_walltime"] and status["electronic_converged"] is False
    assert status["band_converged"] is False and status["partial_results"]
    assert status["replica_logs_read"] == ["aiida-BAND1.out", "aiida-BAND2.out"]
    good = parse_neb_status(logs(), 2)
    assert good["electronic_converged"] and good["band_converged"]
    assert not good["partial_results"]
    timed = parse_neb_status(logs(timeout=True), 2)
    assert timed["band_converged"] is True and timed["exceeded_walltime"]
    assert parse_neb_status({"aiida.out": ""}, 2)["band_converged"] is None


def parser_node(computer, periodic, timeout=False, restart=True):
    params = {
        "FORCE_EVAL": {
            "SUBSYS": {"CELL": {"PERIODIC": periodic}},
            "DFT": {
                "POISSON": {
                    "PERIODIC": periodic,
                    "POISSON_SOLVER": solver_for(periodic),
                }
            },
        },
        "MOTION": {"BAND": {"NUMBER_OF_REPLICA": 2}},
    }
    node = orm.CalcJobNode(computer=computer, process_type="aiida.calculations:cp2k")
    node.set_option("parser_name", "nanotech_empa.cp2k_neb_parser")
    node.base.links.add_incoming(
        orm.Dict(params).store(), link_type=LinkType.INPUT_CALC, link_label="parameters"
    )
    node.store()
    folder = orm.FolderData()
    texts = logs(timeout=timeout)
    if restart:
        numeric = "\n &COORD\n 5 6 6\n 6 6 6\n &END COORD\n"
        texts["aiida-1.restart"] = (
            "\n &CELL\n A 12 0 0\n B 0 12 0\n C 0 0 12\n &END CELL\n"
            + numeric * 2
            + "\n &COORD\n H 5 6 6\n H 6 6 6\n &END COORD\n"
        )
    for name, text in texts.items():
        folder.base.repository.put_object_from_filelike(io.StringIO(text), name)
    folder.base.links.add_incoming(
        node, link_type=LinkType.CREATE, link_label="retrieved"
    )
    folder.store()
    return node


@pytest.mark.parametrize("periodic", PERIODICITIES)
def test_partial_parser_preserves_arrays_and_effective_pbc(
    aiida_profile_clean, aiida_localhost, monkeypatch, periodic
):
    parser = Cp2kNebParser(parser_node(aiida_localhost, periodic, timeout=True))
    result = parser.parse()
    assert result.status == 400
    assert parser.outputs["output_parameters"]["exceeded_walltime"]
    assert parser.outputs["replica_energies"].get_array("energies").shape == (1, 2)
    assert parser.outputs["opt_replica_000"].pbc == periodic_pbc(periodic)
    captured = {}
    harness = SimpleNamespace(
        ctx=SimpleNamespace(
            neb=SimpleNamespace(
                is_finished_ok=False, outputs=AttributeDict(parser.outputs)
            )
        ),
        inputs=SimpleNamespace(structure=object()),
        node=SimpleNamespace(uuid="synthetic-only"),
        report=lambda text: None,
        out=lambda label, value: captured.update({label: value}),
        exit_codes=Cp2kNebWorkChain.spec().exit_codes,
    )
    monkeypatch.setattr(
        "aiida_nanotech_empa.utils.common_utils.add_extras", lambda *args: None
    )
    assert Cp2kNebWorkChain.finalize(harness).status == 391
    assert set(parser.outputs) <= set(captured)


@pytest.mark.parametrize("timeout", [False, True])
def test_missing_restart_keeps_energy_arrays(
    aiida_profile_clean, aiida_localhost, timeout
):
    parser = Cp2kNebParser(
        parser_node(aiida_localhost, "XYZ", timeout=timeout, restart=False)
    )
    assert parser.parse().status == (400 if timeout else 312)
    assert parser.outputs["replica_energies"].get_array("energies").shape == (1, 2)
