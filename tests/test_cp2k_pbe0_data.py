"""Check every advertised PBE0 default against the actual bundled data."""

import functools
import re

import pytest
import yaml
from ase.data import atomic_numbers, chemical_symbols

from aiida_nanotech_empa.workflows.cp2k import cp2k_utils

DATA_DIR = cp2k_utils.DATA_DIR
MAPS = yaml.safe_load((DATA_DIR / "atomic_kinds.yml").read_text())
LANTHANIDES = chemical_symbols[57:72]


@functools.cache
def _definitions(filename):
    """Read CP2K definition headers, including all aliases on each header."""
    lines = [
        line.split("#", 1)[0].split()
        for line in (DATA_DIR / filename).read_text().splitlines()
    ]
    lines = [tokens for tokens in lines if tokens]
    entries = {}
    for index, tokens in enumerate(lines):
        if len(tokens) > 1 and tokens[0] in atomic_numbers:
            for alias in tokens[1:]:
                entries[tokens[0], alias] = lines[index + 1]
    return entries


def _kind(element, params):
    return cp2k_utils.get_kinds_section(
        {element: {"mag": 0.0, "ghost": 0}}, "gpw", params
    )["FORCE_EVAL"]["SUBSYS"]["KIND"][0]


def test_pbe0_maps_cover_the_same_supported_elements():
    elements = set(MAPS["pbe0_basis_set"])
    assert len(elements) == 71
    assert elements == set(MAPS["pbe0_aux_basis_set"])
    assert elements == set(MAPS["pbe0_pseudopotential"])
    assert not elements.intersection(LANTHANIDES)


@pytest.mark.parametrize("element", sorted(MAPS["pbe0_basis_set"]))
def test_pbe0_default_definitions_exist_with_matching_valence(element):
    kind = _kind(element, {"xc_functional": "PBE0"})
    basis = kind["BASIS_SET"]
    auxiliary = kind["BASIS_SET AUX_FIT"]
    potential = kind["POTENTIAL"]
    assert "BASIS_SET RI_AUX" not in kind
    assert (element, basis) in _definitions("BASIS_MOLOPT_UZH")
    assert (element, auxiliary) in _definitions("BASIS_ADMM_UZH")
    assert (element, potential) in _definitions("POTENTIAL_UZH")
    valences = {
        int(re.search(r"-q(\d+)$", value).group(1))
        for value in (basis, auxiliary, potential)
    }
    assert len(valences) == 1
    electron_counts = _definitions("POTENTIAL_UZH")[element, potential]
    assert sum(map(int, electron_counts)) == valences.pop()


@pytest.mark.parametrize(("element", "charge"), [("Na", 9), ("Mg", 10), ("Rh", 17)])
def test_approved_semicore_defaults(element, charge):
    kind = _kind(element, {"xc_functional": "PBE0"})
    assert kind["BASIS_SET"] == f"TZVP-MOLOPT-PBE0-GTH-q{charge}"
    assert kind["BASIS_SET AUX_FIT"] == f"admm-tzp-q{charge}"
    assert kind["POTENTIAL"] == f"GTH-PBE0-q{charge}"


@pytest.mark.parametrize("element", LANTHANIDES)
def test_unsupported_lanthanide_defaults_fail_early(element):
    with pytest.raises(ValueError, match=rf"No default.*{element}") as error:
        _kind(element, {"xc_functional": "PBE0"})
    assert "override" in str(error.value)
    assert "matching data files" in str(error.value)


def test_explicit_custom_overrides_can_supply_an_unsupported_element():
    # This checks forwarding only; actual custom data need scientific validation.
    kind = _kind(
        "La",
        {
            "xc_functional": "PBE0",
            "basis_set_overrides": {"La": "custom-orbital"},
            "aux_basis_set_overrides": {"La": "custom-auxiliary"},
            "potential_overrides": {"La": "custom-potential"},
        },
    )
    assert kind["BASIS_SET"] == "custom-orbital"
    assert kind["BASIS_SET AUX_FIT"] == "custom-auxiliary"
    assert kind["POTENTIAL"] == "custom-potential"


@pytest.mark.parametrize("params", [{"xc_functional": "PBE0"}, {"functional": "pbe0"}])
def test_pbe0_stages_both_bases_potential_and_coulomb_data(params, aiida_profile):
    files = cp2k_utils.get_dft_file_inputs(params)
    assert {key: node.filename for key, node in files.items()} == {
        "basis": "BASIS_MOLOPT_UZH",
        "basis_2": "BASIS_ADMM_UZH",
        "pseudo": "POTENTIAL_UZH",
        "extra_1": "t_c_g.dat",
    }
    parameters = cp2k_utils.load_protocol("scf_ot_protocol.yml", "standard")
    cp2k_utils.apply_dft_file_names(parameters, params)
    cp2k_utils.apply_xc_settings(parameters, params)
    dft = parameters["FORCE_EVAL"]["DFT"]
    assert dft["BASIS_SET_FILE_NAME"] == ["BASIS_MOLOPT_UZH", "BASIS_ADMM_UZH"]
    assert dft["POTENTIAL_FILE_NAME"] == files["pseudo"].filename
    assert (
        dft["XC"]["HF"]["INTERACTION_POTENTIAL"]["T_C_G_DATA"]
        == files["extra_1"].filename
    )
