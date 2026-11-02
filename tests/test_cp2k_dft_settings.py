"""DFT settings must survive staging and every calculation submission."""

import copy

import ase
import pytest
from aiida import engine, orm, plugins
from aiida.common.links import LinkType
from aiida.manage import get_manager

from aiida_nanotech_empa.workflows.cp2k import cp2k_utils


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({}, ["BASIS_MOLOPT"]),
        ({"basis_set_file_name": "BASIS_DEFAULT"}, ["BASIS_DEFAULT"]),
        (
            {
                "basis_set_file_name": "BASIS_DEFAULT",
                "basis_set_file_names": ["BASIS_MOLOPT", "BASIS_DEFAULT"],
            },
            ["BASIS_MOLOPT", "BASIS_DEFAULT"],
        ),
    ],
)
def test_basis_file_names_match_staged_files(params, expected, aiida_profile):
    files = cp2k_utils.get_dft_file_inputs(params)
    parameters = {"FORCE_EVAL": {"DFT": {}}}
    cp2k_utils.apply_dft_file_names(parameters, params)
    dft = parameters["FORCE_EVAL"]["DFT"]

    assert [node.filename for key, node in files.items() if key != "pseudo"] == expected
    assert dft["BASIS_SET_FILE_NAME"] == (
        expected[0] if len(expected) == 1 else expected
    )
    assert dft["POTENTIAL_FILE_NAME"] == files["pseudo"].filename == "POTENTIAL"
    for node in files.values():
        assert node.get_content() == (cp2k_utils.DATA_DIR / node.filename).read_text()


def test_kind_overrides_keep_other_elements_spin_and_ghost_flags():
    kinds = {"C1": {"mag": 1.0, "ghost": 0}, "H1": {"mag": 0.0, "ghost": 1}}
    original = cp2k_utils.get_kinds_section(kinds, "gpw")["FORCE_EVAL"]["SUBSYS"][
        "KIND"
    ]
    params = {
        "basis_set_overrides": {"C": "DZVP-MOLOPT-GTH"},
        "potential_overrides": {"C": "GTH-PBE"},
    }
    updated = cp2k_utils.get_kinds_section(kinds, "gpw", params)["FORCE_EVAL"][
        "SUBSYS"
    ]["KIND"]

    assert updated[0] == {
        **original[0],
        "BASIS_SET": "DZVP-MOLOPT-GTH",
        "POTENTIAL": "GTH-PBE",
    }
    assert updated[0]["MAGNETIZATION"] == 1.0
    assert updated[1] == original[1]
    assert updated[1]["GHOST"] == "TRUE"


def test_kind_mapping_keys_enable_an_auxiliary_basis():
    section = cp2k_utils.get_kinds_section(
        {"C": {"mag": 0.0, "ghost": 0}},
        "gpw",
        {
            "basis_set_key": "gpw_std_gw_basis_set",
            "aux_basis_set_key": "gpw_std_gw_basis_set_aux",
            "potential_key": "all",
            "aux_basis_set_overrides": {"C": "RI_aug_TZ"},
        },
    )["FORCE_EVAL"]["SUBSYS"]["KIND"][0]
    assert section["BASIS_SET"] == "aug-DZVP-GTH-up-up-up-up"
    assert section["BASIS_SET RI_AUX"] == "RI_aug_TZ"
    assert section["POTENTIAL"] == "ALL"


@pytest.mark.parametrize("functional", [None, "PBE", "PBE-D3", "PBE_D3"])
def test_pbe_xc_settings_preserve_the_protocol(functional):
    parameters = cp2k_utils.load_protocol("geo_opt_protocol.yml", "standard")
    original = copy.deepcopy(parameters)
    params = {} if functional is None else {"xc_functional": functional}
    cp2k_utils.apply_xc_settings(parameters, params)
    assert parameters == original


@pytest.mark.parametrize("functional", ["PBE", "PBE0"])
def test_ot_minimizer_override(functional):
    parameters = cp2k_utils.load_protocol("scf_ot_protocol.yml", "standard")
    params = {"xc_functional": functional, "ot_minimizer": "DIIS"}
    cp2k_utils.apply_xc_settings(parameters, params)
    assert parameters["FORCE_EVAL"]["DFT"]["SCF"]["OT"]["MINIMIZER"] == "DIIS"


def test_pbe0_tunes_qs_and_ot_scf_and_keeps_the_rest():
    parameters = cp2k_utils.load_protocol("scf_ot_protocol.yml", "standard")
    original = copy.deepcopy(parameters["FORCE_EVAL"]["DFT"])
    cp2k_utils.apply_xc_settings(parameters, {"xc_functional": "PBE0", "max_scf": 7})
    dft = parameters["FORCE_EVAL"]["DFT"]
    assert dft["QS"] == {"METHOD": "GPW", "EPS_PGF_ORB": 1e-32}
    assert dft["SCF"]["MAX_SCF"] == 7
    assert dft["SCF"]["EPS_LUMO"] == original["SCF"]["EPS_SCF"]
    assert dft["SCF"]["OT"]["PRECONDITIONER"] == "FULL_KINETIC"
    assert dft["SCF"]["OUTER_SCF"]["MAX_SCF"] == 100
    assert dft["SCF"]["PRINT"] == original["SCF"]["PRINT"]
    pair_potential = dft["XC"]["VDW_POTENTIAL"]["PAIR_POTENTIAL"]
    assert pair_potential["REFERENCE_FUNCTIONAL"] == "PBE0"


def test_unknown_xc_functional_fails_explicitly():
    parameters = cp2k_utils.load_protocol("geo_opt_protocol.yml", "standard")
    with pytest.raises(ValueError, match="Unsupported CP2K XC functional: SCAN"):
        cp2k_utils.apply_xc_settings(parameters, {"xc_functional": "SCAN"})


@pytest.mark.parametrize("workflow", ["geo_opt", "diag", "scf"])
@pytest.mark.parametrize(
    "custom", [False, True], ids=["defaults", "custom-files-and-kind"]
)
def test_submitted_dft_settings_reach_every_stage(
    fixture_localhost, monkeypatch, workflow, custom
):
    code = orm.InstalledCode(
        computer=fixture_localhost,
        filepath_executable="/bin/true",
        default_calc_job_plugin="cp2k",
    ).store()
    structure = orm.StructureData(
        ase=ase.Atoms(
            "H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[12, 12, 12], pbc=True
        )
    )
    params = {}
    if custom:
        params = {
            "basis_set_file_names": ["BASIS_MOLOPT", "BASIS_DEFAULT"],
            "potential_file_name": "POTENTIALS_DEFAULT",
            "basis_set_overrides": {"H": "SZV-MOLOPT-SR-GTH-q1"},
            "potential_overrides": {"H": "GTH-PBE"},
            "periodic": "NONE",
            "poisson_solver": "MT",
            "poisson_settings": {"MT": {"ALPHA": 9.0}},
        }
        from aiida_nanotech_empa.workflows.cp2k.electrostatics import prepare_cell

        structure = orm.StructureData(ase=prepare_cell(structure.get_ase(), params))
    workchain = plugins.WorkflowFactory(f"nanotech_empa.cp2k.{workflow}")
    inputs = {"structure": structure, "dft_params": orm.Dict(params)}
    if workflow == "geo_opt":
        inputs.update(
            code=code,
            sys_params=orm.Dict({}),
            options={"max_wallclock_seconds": 1200, "resources": {"num_machines": 1}},
        )
    else:
        inputs["cp2k_code"] = code
    process = get_manager().get_runner().instantiate_process(workchain, **inputs)
    submitted = []
    children = []

    def submit(_, builder):
        submitted.append(builder)
        child = orm.WorkChainNode().store()
        child.set_process_state(engine.ProcessState.FINISHED)
        child.set_exit_status(0)
        remote = orm.RemoteData(
            computer=fixture_localhost, remote_path="/tmp/dft-settings-test"
        ).store()
        remote.base.links.add_incoming(
            child, link_type=LinkType.RETURN, link_label="remote_folder"
        )
        children.append(child)
        return child

    monkeypatch.setattr(workchain, "submit", submit)
    try:
        process.setup()
        if workflow == "geo_opt":
            process.submit_calc()
        else:
            process.run_ot_scf()
            process.ctx.ot_scf = children[-1]
            process.run_diag_scf()
        assert len(submitted) == (1 if workflow == "geo_opt" else 2)
        for index, builder in enumerate(submitted):
            dft = builder.cp2k.parameters.get_dict()["FORCE_EVAL"]["DFT"]
            assert dft["POISSON"] == (
                {"PERIODIC": "NONE", "POISSON_SOLVER": "MT", "MT": {"ALPHA": 9.0}}
                if custom
                else {"PERIODIC": "XYZ", "POISSON_SOLVER": "PERIODIC"}
            )
            assert builder.cp2k.structure.cell == structure.cell
            assert builder.cp2k.structure.get_ase() == structure.get_ase()
            if workflow == "geo_opt" or (workflow == "scf" and index == 0):
                assert dft["PRINT"]["LOWDIN"] == {"PRINT_GOP": "T"}
            else:
                assert "LOWDIN" not in dft.get("PRINT", {})
            kind = builder.cp2k.parameters.get_dict()["FORCE_EVAL"]["SUBSYS"]["KIND"][0]
            assert dft["BASIS_SET_FILE_NAME"] == (
                ["BASIS_MOLOPT", "BASIS_DEFAULT"] if custom else "BASIS_MOLOPT"
            )
            assert dft["POTENTIAL_FILE_NAME"] == (
                "POTENTIALS_DEFAULT" if custom else "POTENTIAL"
            )
            assert kind["BASIS_SET"] == (
                "SZV-MOLOPT-SR-GTH-q1" if custom else "TZV2P-MOLOPT-GTH"
            )
            assert kind["POTENTIAL"] == ("GTH-PBE" if custom else "GTH-PBE-q1")
            assert "HF" not in dft["XC"]
            assert {node.filename for node in builder.cp2k.file.values()} == (
                {"BASIS_MOLOPT", "BASIS_DEFAULT", "POTENTIALS_DEFAULT"}
                if custom
                else {"BASIS_MOLOPT", "POTENTIAL"}
            )
    finally:
        process.close()


@pytest.mark.parametrize("functional", ["PBE", "PBE0"])
def test_ot_diis_alias_preserves_unrelated_settings(functional):
    parameters = cp2k_utils.load_protocol("scf_ot_protocol.yml", "standard")
    cp2k_utils.apply_xc_settings(
        parameters, {"xc_functional": functional, "ot_diis": True}
    )
    assert parameters["FORCE_EVAL"]["DFT"]["SCF"]["OT"]["MINIMIZER"] == "DIIS"


def test_advanced_pbe0_controls_preserve_existing_scf_fields():
    parameters = cp2k_utils.load_protocol("scf_ot_protocol.yml", "standard")
    scf = parameters["FORCE_EVAL"]["DFT"]["SCF"]
    scf["OT"]["STEPSIZE"] = 0.08
    original = copy.deepcopy(scf)
    cp2k_utils.apply_xc_settings(
        parameters,
        {
            "xc_functional": "PBE0",
            "eps_scf": 2e-8,
            "outer_scf_eps_scf": 3e-8,
            "scf_guess": "ATOMIC",
        },
    )
    assert parameters["FORCE_EVAL"]["DFT"]["SCF"] is scf
    assert scf["EPS_SCF"] == scf["EPS_LUMO"] == 2e-8
    assert scf["OUTER_SCF"]["EPS_SCF"] == 3e-8
    assert scf["SCF_GUESS"] == "ATOMIC"
    assert scf["OT"]["STEPSIZE"] == original["OT"]["STEPSIZE"]
    assert scf["PRINT"] == original["PRINT"]


@pytest.mark.parametrize(
    "filename",
    ["t_c_g.dat", "coulomb data.dat", '"coulomb data.dat"', "'coulomb data.dat'"],
)
def test_coulomb_data_filename_remains_one_cp2k_token(filename):
    import shlex

    parameters = cp2k_utils.load_protocol("scf_ot_protocol.yml", "standard")
    cp2k_utils.apply_xc_settings(
        parameters, {"xc_functional": "PBE0", "tcg_data_file_name": filename}
    )
    token = parameters["FORCE_EVAL"]["DFT"]["XC"]["HF"]["INTERACTION_POTENTIAL"][
        "T_C_G_DATA"
    ]
    assert shlex.split(token) == [filename.strip("\"'")]
