from types import SimpleNamespace

import pytest
from aiida import orm, plugins

Cp2kScfWorkChain = plugins.WorkflowFactory("nanotech_empa.cp2k.scf")


@pytest.mark.parametrize(
    ("overlap_matrix", "rejected"),
    [
        ("remote_and_sparse_retrieved", "sparse_overlap_code"),
        ("dense", "overlap_matrix"),
    ],
)
def test_validator_overlap_matrix(overlap_matrix, rejected):
    message = Cp2kScfWorkChain._validate_inputs(
        {"overlap_matrix": orm.Str(overlap_matrix)}, None
    )

    assert rejected in message


@pytest.mark.parametrize(
    ("run_diag_scf", "dft_params", "with_settings", "rejected"),
    [
        # Diagonalization skipped: diag-only inputs are rejected.
        (
            False,
            {"nhomo": 3, "added_mos": 10},
            True,
            ["nhomo", "added_mos", "settings"],
        ),
        # Diagonalization runs: diag-only inputs are accepted.
        (True, {"nhomo": 3, "added_mos": 10}, True, None),
        # Falsy diag-only params have no effect, so they are accepted.
        (False, {"sc_diag": False, "elpa_switch": False}, False, None),
    ],
)
def test_validator_diag_only_inputs(run_diag_scf, dft_params, with_settings, rejected):
    inputs = {
        "run_diag_scf": orm.Bool(run_diag_scf),
        "overlap_matrix": orm.Str("none"),
        "dft_params": orm.Dict(dft_params),
    }
    if with_settings:
        inputs["settings"] = orm.Dict()

    message = Cp2kScfWorkChain._validate_inputs(inputs, None)

    if rejected is None:
        assert message is None
    else:
        for name in rejected:
            assert name in message


@pytest.mark.parametrize(
    ("hook", "run_diag_scf", "overlap_matrix", "printed"),
    [
        # The overlap matrix is printed in the last SCF step only.
        ("update_ot_input_dict", False, "remote_only", True),
        ("update_ot_input_dict", True, "remote_only", False),
        ("update_diag_input_dict", True, "remote_only", True),
        ("update_ot_input_dict", False, "none", False),
        ("update_diag_input_dict", True, "none", False),
    ],
)
def test_overlap_matrix_printed_in_last_scf_step(
    hook, run_diag_scf, overlap_matrix, printed
):
    workchain = SimpleNamespace(
        inputs=SimpleNamespace(
            run_diag_scf=orm.Bool(run_diag_scf),
            overlap_matrix=orm.Str(overlap_matrix),
            overlap_ndigits=orm.Int(10),
        ),
        should_run_bader=lambda: False,
    )
    input_dict = {"FORCE_EVAL": {"DFT": {}}}

    getattr(Cp2kScfWorkChain, hook)(workchain, input_dict)

    if printed:
        ao_matrices = input_dict["FORCE_EVAL"]["DFT"]["PRINT"]["AO_MATRICES"]
        assert ao_matrices["OVERLAP"] == "T"
        assert ao_matrices["NDIGITS"] == 10
    else:
        assert "PRINT" not in input_dict["FORCE_EVAL"]["DFT"]
