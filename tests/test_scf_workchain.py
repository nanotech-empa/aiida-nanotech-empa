from types import SimpleNamespace

import pytest
from aiida import orm, plugins

Cp2kScfWorkChain = plugins.WorkflowFactory("nanotech_empa.cp2k.scf")


def test_retrieve_sparse_overlap_requires_sparse_overlap_code():
    message = Cp2kScfWorkChain._validate_inputs(
        {"retrieve_sparse_overlap": orm.Bool(True)}, None
    )

    assert "sparse_overlap_code" in message


def _fake_workchain(write=False, retrieve=False, ndigits=14, dft_params=None):
    return SimpleNamespace(
        inputs=SimpleNamespace(
            write_overlap_matrix=orm.Bool(write),
            retrieve_sparse_overlap=orm.Bool(retrieve),
            overlap_ndigits=orm.Int(ndigits),
            dft_params=orm.Dict(dft_params or {}),
        )
    )


@pytest.mark.parametrize(
    ("write", "retrieve", "dft_params", "expected"),
    [
        (False, False, {}, False),
        (True, False, {}, True),
        (False, True, {}, True),
        (False, False, {"added_mos": 10}, True),
        (False, False, {"added_mos": 0}, False),
    ],
)
def test_should_run_diag_scf(write, retrieve, dft_params, expected):
    workchain = _fake_workchain(write=write, retrieve=retrieve, dft_params=dft_params)

    assert Cp2kScfWorkChain.should_run_diag_scf(workchain) is expected


def test_update_diag_input_dict_adds_ao_matrices():
    input_dict = {"FORCE_EVAL": {"DFT": {}}}
    Cp2kScfWorkChain.update_diag_input_dict(
        _fake_workchain(write=True, ndigits=10), input_dict
    )

    ao_matrices = input_dict["FORCE_EVAL"]["DFT"]["PRINT"]["AO_MATRICES"]
    assert ao_matrices["OVERLAP"] == "T"
    assert ao_matrices["NDIGITS"] == 10


def test_update_diag_input_dict_noop_without_overlap_flags():
    input_dict = {"FORCE_EVAL": {"DFT": {}}}
    Cp2kScfWorkChain.update_diag_input_dict(_fake_workchain(), input_dict)

    assert "PRINT" not in input_dict["FORCE_EVAL"]["DFT"]
