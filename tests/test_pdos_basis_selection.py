"""Select the actual orbital file without changing the SCF basis or its order."""

import pathlib
import shutil
from types import SimpleNamespace

import ase
import pytest
from aiida import orm
from aiida.common import AttributeDict

from aiida_nanotech_empa.workflows.cp2k import cp2k_utils, pdos_workchain


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({}, "BASIS_MOLOPT"),
        ({"xc_functional": "PBE0"}, "BASIS_MOLOPT_UZH"),
        (
            {
                "xc_functional": "PBE0",
                "basis_set_file_names": ["BASIS_ADMM_UZH", "BASIS_MOLOPT_UZH"],
            },
            "BASIS_MOLOPT_UZH",
        ),
        (
            {
                "xc_functional": "PBE0",
                "basis_set_file_names": [
                    "BASIS_ADMM_UZH",
                    "BASIS_MOLOPT",
                    "BASIS_MOLOPT_UZH",
                ],
            },
            "BASIS_MOLOPT_UZH",
        ),
    ],
)
def test_orbital_selection_ignores_auxiliary_and_unrelated_files(params, expected):
    assert cp2k_utils.get_orbital_basis_file_names(params, {"C", "H"}) == [expected]


@pytest.fixture
def custom_basis_files(tmp_path, monkeypatch):
    shutil.copyfile(
        cp2k_utils.DATA_DIR / "atomic_kinds.yml", tmp_path / "atomic_kinds.yml"
    )
    # Minimal named Gaussian definitions: these tests check lookup, not convergence.
    hydrogen = "H chosen-H H-alias\n 1\n 1 0 0 1 1\n 1.0 1.0\n"
    carbon = "C chosen-C C-alias\n 1\n 1 0 0 1 1\n 1.0 1.0\n"
    (tmp_path / "hydrogen").write_text(hydrogen)
    (tmp_path / "carbon").write_text(carbon)
    (tmp_path / "both").write_text(hydrogen + carbon)
    monkeypatch.setattr(cp2k_utils, "DATA_DIR", tmp_path)
    return {"basis_set_overrides": {"H": "H-alias", "C": "C-alias"}}


def test_orbital_selection_accepts_a_complete_custom_file(custom_basis_files):
    params = {**custom_basis_files, "basis_set_file_names": ["both"]}
    assert cp2k_utils.get_orbital_basis_file_names(params, {"H", "C"}) == ["both"]


@pytest.mark.parametrize("files", [["hydrogen", "carbon"], ["hydrogen", "both"]])
def test_orbital_selection_keeps_all_required_files(custom_basis_files, files):
    params = {**custom_basis_files, "basis_set_file_names": files}
    assert cp2k_utils.get_orbital_basis_file_names(params, {"H", "C"}) == files


def test_orbital_selection_respects_first_file_precedence(custom_basis_files):
    params = {**custom_basis_files, "basis_set_file_names": ["both", "hydrogen"]}
    assert cp2k_utils.get_orbital_basis_file_names(params, {"H", "C"}) == ["both"]


def test_orbital_selection_reports_a_missing_definition(custom_basis_files):
    params = {**custom_basis_files, "basis_set_file_names": ["hydrogen"]}
    with pytest.raises(ValueError, match="C/C-alias"):
        cp2k_utils.get_orbital_basis_file_names(params, {"H", "C"})


@pytest.mark.parametrize("overlap_enabled", [False, True])
def test_setup_checks_overlap_support_before_starting_scf(
    aiida_localhost, monkeypatch, overlap_enabled
):
    structure = orm.StructureData(
        ase=ase.Atoms(
            "H4",
            positions=[[0, 0, 0], [0, 0, 0.74], [0, 0, 4], [0, 0, 4.74]],
            cell=[12] * 3,
        )
    )
    code = orm.InstalledCode(
        computer=aiida_localhost,
        filepath_executable="/bin/true",
        default_calc_job_plugin="nanotech_empa.overlap",
    ).store()
    process = SimpleNamespace(
        inputs=AttributeDict(
            {
                "structure": structure,
                "molecule_indices": orm.List([0, 1]),
                "dft_params": orm.Dict({"charges": {"all": 0, "molecule": 0}}),
                "overlap_params": orm.Dict({"--emax1": "3", "--nlumo2": "2"}),
                "do_overlap": orm.Bool(overlap_enabled),
                "overlap_code": code,
            }
        ),
        ctx=SimpleNamespace(),
        report=lambda message: None,
    )
    calls = []

    def reject(params, elements):
        calls.append(elements)
        raise ValueError("missing selected orbital basis")

    monkeypatch.setattr(cp2k_utils, "get_orbital_basis_file_names", reject)
    if overlap_enabled:
        with pytest.raises(ValueError, match="missing selected orbital basis"):
            pdos_workchain.Cp2kPdosWorkChain.setup(process)
        assert calls == [{"H"}]
    else:
        pdos_workchain.Cp2kPdosWorkChain.setup(process)
        assert calls == []


def test_overlap_forwards_each_parents_actual_orbital_files(
    aiida_localhost, custom_basis_files
):
    code = orm.InstalledCode(
        computer=aiida_localhost,
        filepath_executable="/bin/true",
        default_calc_job_plugin="nanotech_empa.overlap",
    ).store()
    params = {**custom_basis_files, "basis_set_file_names": ["hydrogen", "carbon"]}
    slab = orm.RemoteData(computer=aiida_localhost, remote_path="/tmp/slab")
    molecule = orm.RemoteData(computer=aiida_localhost, remote_path="/tmp/molecule")
    submitted = []
    process = SimpleNamespace(
        inputs=AttributeDict(
            {
                "overlap_code": code,
                "cp2k_code": code,
                "overlap_params": orm.Dict({"--emax": "3"}),
            }
        ),
        ctx=SimpleNamespace(
            dft_parameters=params,
            mol_dft_parameters=params,
            structure=orm.StructureData(
                ase=ase.Atoms("CH", positions=[[0, 0, 0], [0, 0, 1]], cell=[10] * 3)
            ),
            molecule_structure=orm.StructureData(
                ase=ase.Atoms("H", positions=[[0, 0, 1]], cell=[10] * 3)
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
    pdos_workchain.Cp2kPdosWorkChain.run_overlap(process)
    actual = submitted[0].parameters.get_dict()
    assert actual == {
        "--basis_set_file1": [
            "parent_slab_folder/hydrogen",
            "parent_slab_folder/carbon",
        ],
        "--basis_set_file2": "parent_mol_folder/hydrogen",
        "--emax": "3",
    }


def test_overlap_calcjob_expands_basis_file_lists(aiida_localhost):
    from aiida.common.folders import SandboxFolder
    from aiida.manage import get_manager

    from aiida_nanotech_empa.plugins.overlap import OverlapCalculation

    code = orm.InstalledCode(
        computer=aiida_localhost,
        filepath_executable="/bin/true",
        default_calc_job_plugin="nanotech_empa.overlap",
    ).store()
    process = (
        get_manager()
        .get_runner()
        .instantiate_process(
            OverlapCalculation,
            code=code,
            parameters=orm.Dict(
                {
                    "--basis_set_file1": [
                        "parent_slab_folder/a",
                        "parent_slab_folder/b",
                    ],
                    "--basis_set_file2": "parent_mol_folder/c",
                }
            ),
            settings=orm.Dict({}),
            parent_slab_folder=orm.RemoteData(
                computer=aiida_localhost, remote_path="/tmp/slab"
            ),
            parent_mol_folder=orm.RemoteData(
                computer=aiida_localhost, remote_path="/tmp/molecule"
            ),
            metadata={
                "options": {
                    "resources": {"num_machines": 1, "num_mpiprocs_per_machine": 1}
                }
            },
        )
    )
    try:
        with SandboxFolder() as folder:
            info = process.prepare_for_submission(folder)
        assert info.codes_info[0].cmdline_params == [
            "--basis_set_file1",
            "parent_slab_folder/a",
            "parent_slab_folder/b",
            "--basis_set_file2",
            "parent_mol_folder/c",
        ]
    finally:
        process.close()


@pytest.mark.parametrize("workflow", ["stm", "orbitals", "hrstm"])
@pytest.mark.parametrize("selection", ["pbe", "pbe0-aux-first", "split", "explicit"])
def test_stm_workflows_submit_actual_orbital_files(
    fixture_localhost, custom_basis_files, workflow, selection, monkeypatch
):
    from aiida import plugins
    from aiida.common.folders import SandboxFolder
    from aiida.manage import get_manager

    # Restore the packaged data for default PBE/PBE0; use split fixtures otherwise.
    params = {**custom_basis_files, "basis_set_file_names": ["hydrogen", "carbon"]}
    expected = ["parent_calc_folder/hydrogen", "parent_calc_folder/carbon"]
    explicit = "parent_calc_folder/BASIS_MOLOPT"
    if selection in ("pbe", "pbe0-aux-first"):
        monkeypatch.setattr(
            cp2k_utils, "DATA_DIR", pathlib.Path(cp2k_utils.__file__).parent / "data"
        )
        params = (
            {}
            if selection == "pbe"
            else {
                "xc_functional": "PBE0",
                "basis_set_file_names": ["BASIS_ADMM_UZH", "BASIS_MOLOPT_UZH"],
            }
        )
        expected = "parent_calc_folder/" + (
            "BASIS_MOLOPT" if selection == "pbe" else "BASIS_MOLOPT_UZH"
        )
    elif selection == "explicit":
        explicit = ["user/orbitals-C", "user/orbitals-H"]
        expected = explicit
        # An explicit remote override must not try to resolve packaged files.
        params = {"basis_set_file_names": ["not-a-packaged-file"]}

    code = orm.InstalledCode(
        computer=fixture_localhost,
        filepath_executable="/bin/true",
        default_calc_job_plugin="nanotech_empa.hrstm"
        if workflow == "hrstm"
        else "nanotech_empa.stm",
    ).store()
    remote = orm.RemoteData(computer=fixture_localhost, remote_path="/tmp/parent")
    structure = orm.StructureData(
        ase=ase.Atoms("CH", positions=[[0, 0, 0], [0, 0, 1]], cell=[10] * 3)
    )
    submitted = []
    chain = SimpleNamespace(
        inputs=AttributeDict(
            {
                "structure": structure,
                "cp2k_code": code,
                "spm_code": code,
                "hrstm_code": code,
                "protocol": orm.Str("standard"),
                "spm_params": orm.Dict({"--basis_set_file": explicit}),
                "hrstm_params": orm.Dict({"--basis_set_file": explicit}),
            }
        ),
        ctx=SimpleNamespace(
            n_atoms=2,
            dft_params=params,
            diag_scf=SimpleNamespace(
                is_finished_ok=True, outputs=SimpleNamespace(remote_folder=remote)
            ),
            ppm=SimpleNamespace(
                outputs=SimpleNamespace(
                    remote_folder=remote,
                    retrieved=SimpleNamespace(
                        list_objects=lambda: [
                            SimpleNamespace(name=name)
                            for name in ["df.npy", "df_vec.npy"]
                        ]
                    ),
                )
            ),
        ),
        report=lambda message: None,
        submit=lambda cls, **inputs: submitted.append((cls, inputs)),
    )
    workchain = plugins.WorkflowFactory(f"nanotech_empa.cp2k.{workflow}")
    getattr(workchain, "run_hrstm" if workflow == "hrstm" else "run_stm")(chain)
    cls, inputs = submitted[0]
    assert inputs["parameters"].get_dict()["--basis_set_file"] == expected
    process = get_manager().get_runner().instantiate_process(cls, **inputs)
    try:
        with SandboxFolder() as folder:
            info = process.prepare_for_submission(folder)
        paths = expected if isinstance(expected, list) else [expected]
        assert info.codes_info[0].cmdline_params == ["--basis_set_file", *paths]
    finally:
        process.close()


def test_generated_basis_parameter_handles_omitted_flag_and_does_not_mutate_input():
    original = {"--energy_range": ["-2", "2", "0.1"]}
    result = cp2k_utils.update_legacy_basis_parameter(
        original, {"xc_functional": "PBE0"}, {"H"}
    )
    assert result["--basis_set_file"] == "parent_calc_folder/BASIS_MOLOPT_UZH"
    assert "--basis_set_file" not in original
