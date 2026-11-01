"""Request charge diagnostics without changing the electronic-structure input."""

import copy

import pytest

from aiida_nanotech_empa.workflows.cp2k import cp2k_utils


@pytest.mark.parametrize(
    "existing_print",
    [
        None,
        {"MO_CUBES": {"NHOMO": 3}},
        {"LOWDIN": {"PRINT_GOP": "F", "FILENAME": "custom"}},
    ],
)
def test_default_charge_analysis_preserves_other_parameters(existing_print):
    parameters = cp2k_utils.load_protocol("scf_ot_protocol.yml", "standard")
    dft = parameters["FORCE_EVAL"]["DFT"]
    if existing_print is None:
        dft.pop("PRINT", None)
    else:
        dft["PRINT"] = copy.deepcopy(existing_print)
    original = copy.deepcopy(parameters)
    expected_print = copy.deepcopy(existing_print or {})
    expected_print.setdefault("LOWDIN", {}).setdefault("PRINT_GOP", "T")
    cp2k_utils.apply_default_charge_analysis(parameters)
    assert dft["PRINT"] == expected_print
    cp2k_utils.apply_default_charge_analysis(parameters)
    assert dft["PRINT"] == expected_print
    if existing_print is None:
        dft.pop("PRINT")
    else:
        dft["PRINT"] = existing_print
    assert parameters == original
