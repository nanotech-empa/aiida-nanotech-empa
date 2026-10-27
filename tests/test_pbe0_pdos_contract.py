"""Verify the authored hybrid diagonalization and matching overlap basis inputs."""

import copy
import shutil
from types import SimpleNamespace

import ase
import pytest
from aiida import orm
from aiida.common import AttributeDict

from aiida_nanotech_empa.workflows.cp2k import cp2k_utils
from aiida_nanotech_empa.workflows.cp2k.diag_workchain import Cp2kDiagWorkChain
from aiida_nanotech_empa.workflows.cp2k.pdos_workchain import Cp2kPdosWorkChain


@pytest.mark.parametrize(
    ("method", "purification"), [("ot", "MO_DIAG"), ("diag", "NONE")]
)
def test_pbe0_purification_depends_on_scf_method(method, purification):
    template = cp2k_utils.load_protocol("scf_ot_protocol.yml", "debug")
    cp2k_utils.apply_xc_settings(template, {"xc_functional": "PBE0"}, scf_method=method)
    dft = template["FORCE_EVAL"]["DFT"]
    assert (
        dft["AUXILIARY_DENSITY_MATRIX_METHOD"]["ADMM_PURIFICATION_METHOD"]
        == purification
    )
    assert dft["XC"]["HF"]["FRACTION"] == 0.25


@pytest.mark.parametrize(
    ("override", "purification"), [(None, "NONE"), ("MO_DIAG", "MO_DIAG")]
)
def test_diagonalization_builder_updates_admm_without_mutating_ot(
    aiida_localhost, override, purification
):
    params = {
        "xc_functional": "PBE0",
        "elpa_switch": False,
        "uks": False,
        "sc_diag": True,
        "added_mos": 12,
    }
    if override is not None:
        params["admm_purification_method"] = override
    template = cp2k_utils.load_protocol("scf_ot_protocol.yml", "debug")
    cp2k_utils.apply_xc_settings(template, params)
    original = copy.deepcopy(template)
    code = orm.InstalledCode(
        computer=aiida_localhost,
        filepath_executable="/bin/true",
        default_calc_job_plugin="cp2k",
    ).store()
    submitted = []
    restart_defaults = {
        name: Cp2kDiagWorkChain.spec().inputs[name].default()
        for name in (
            "max_iterations",
            "clean_workdir",
            "on_unhandled_failure",
            "pause_on_max_iterations",
        )
    }
    process = SimpleNamespace(
        inputs=AttributeDict(
            {"cp2k_code": code, "protocol": orm.Str("debug"), **restart_defaults}
        ),
        ctx=SimpleNamespace(
            input_dict=template,
            dft_params=params,
            ot_scf=SimpleNamespace(
                is_finished_ok=True,
                outputs=SimpleNamespace(
                    remote_folder=orm.RemoteData(
                        computer=aiida_localhost, remote_path="/tmp/ot"
                    )
                ),
            ),
            files=cp2k_utils.get_dft_file_inputs(params),
            structure_with_tags=ase.Atoms(
                "H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[8] * 3
            ),
            options={
                "resources": {"num_machines": 1, "num_mpiprocs_per_machine": 1},
                "max_wallclock_seconds": 1800,
            },
        ),
        report=lambda message: None,
        update_diag_input_dict=lambda value: None,
        submit=lambda builder: submitted.append(builder),
        to_context=lambda **kwargs: None,
    )
    Cp2kDiagWorkChain.run_diag_scf(process)
    assert len(submitted) == 1
    dft = submitted[0].cp2k.parameters.get_dict()["FORCE_EVAL"]["DFT"]
    assert (
        dft["AUXILIARY_DENSITY_MATRIX_METHOD"]["ADMM_PURIFICATION_METHOD"]
        == purification
    )
    assert dft["SCF"]["ADDED_MOS"] == 12
    assert dft["XC"]["HF"]["FRACTION"] == 0.25
    assert template == original


@pytest.mark.parametrize(
    ("params", "basis"),
    [
        ({}, "BASIS_MOLOPT"),
        ({"xc_functional": "PBE0"}, "BASIS_MOLOPT_UZH"),
        ({"basis_set_file_names": ["CUSTOM_BASIS", "CUSTOM_ADMM"]}, "CUSTOM_BASIS"),
        pytest.param(
            {
                "xc_functional": "PBE0",
                "basis_set_file_names": ["BASIS_ADMM_UZH", "BASIS_MOLOPT_UZH"],
            },
            "BASIS_MOLOPT_UZH",
            id="reordered-pbe0-files",
        ),
    ],
)
def test_pdos_overlap_uses_scf_basis_for_both_parents(
    aiida_localhost, params, basis, tmp_path, monkeypatch
):
    if basis == "CUSTOM_BASIS":
        data_dir = cp2k_utils.DATA_DIR
        for source, target in [
            ("atomic_kinds.yml", "atomic_kinds.yml"),
            ("BASIS_MOLOPT", "CUSTOM_BASIS"),
            ("BASIS_ADMM_UZH", "CUSTOM_ADMM"),
        ]:
            shutil.copyfile(data_dir / source, tmp_path / target)
        monkeypatch.setattr(cp2k_utils, "DATA_DIR", tmp_path)
    code = orm.InstalledCode(
        computer=aiida_localhost,
        filepath_executable="/bin/true",
        default_calc_job_plugin="nanotech_empa.overlap",
    ).store()
    original = {"--basis_set_file1": "old1", "--basis_set_file2": "old2", "--emax": "3"}
    input_node = orm.Dict(original).store()
    slab = orm.RemoteData(computer=aiida_localhost, remote_path="/tmp/slab")
    molecule = orm.RemoteData(computer=aiida_localhost, remote_path="/tmp/molecule")
    submitted = []
    process = SimpleNamespace(
        inputs=AttributeDict(
            {"overlap_code": code, "cp2k_code": code, "overlap_params": input_node}
        ),
        ctx=SimpleNamespace(
            dft_parameters=params,
            structure=orm.StructureData(
                ase=ase.Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[10] * 3)
            ),
            mol_dft_parameters=params,
            molecule_structure=orm.StructureData(
                ase=ase.Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[10] * 3)
            ),
            n_slab_atoms=2,
            slab_diag_scf=SimpleNamespace(
                is_finished_ok=True, outputs=SimpleNamespace(remote_folder=slab)
            ),
            mol_diag_scf=SimpleNamespace(
                is_finished_ok=True, outputs=SimpleNamespace(remote_folder=molecule)
            ),
        ),
        report=lambda message: None,
        submit=lambda builder: submitted.append(builder),
    )
    Cp2kPdosWorkChain.run_overlap(process)
    assert len(submitted) == 1
    actual = submitted[0].parameters.get_dict()
    assert actual == {
        "--basis_set_file1": f"parent_slab_folder/{basis}",
        "--basis_set_file2": f"parent_mol_folder/{basis}",
        "--emax": "3",
    }
    assert input_node.get_dict() == original
    assert submitted[0].parent_slab_folder == slab
    assert submitted[0].parent_mol_folder == molecule
