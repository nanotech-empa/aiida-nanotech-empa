"""Regression tests for the optional OT-only Bader branch."""

import copy
import io

import ase
import pytest
from aiida import common, engine, orm, plugins
from aiida.common.folders import SandboxFolder
from aiida.common.links import LinkType
from aiida.manage import get_manager

from aiida_nanotech_empa.workflows.cp2k import cp2k_utils

BaderCalculation = plugins.CalculationFactory("nanotech_empa.bader")
BaderParser = plugins.ParserFactory("nanotech_empa.bader")
Cp2kScfWorkChain = plugins.WorkflowFactory("nanotech_empa.cp2k.scf")

PINNED_BADER_OPTIONS = ["-i", "cube", "-b", "neargrid", "-m", "known", "-vac", "off"]


@pytest.fixture
def scf_process(fixture_localhost):
    code = orm.InstalledCode(
        computer=fixture_localhost,
        filepath_executable="/bin/true",
        default_calc_job_plugin="cp2k",
    ).store()
    structure = orm.StructureData(
        ase=ase.Atoms(
            "H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[8, 8, 8], pbc=True
        )
    )
    processes = []

    def create(**kwargs):
        inputs = {
            "cp2k_code": code,
            "structure": structure,
            "dft_params": orm.Dict(dict={}),
        }
        inputs.update(kwargs)
        process = (
            get_manager().get_runner().instantiate_process(Cp2kScfWorkChain, **inputs)
        )
        processes.append(process)
        return process

    yield create
    for process in processes:
        process.close()


@pytest.fixture
def bader_code(fixture_localhost):
    return orm.InstalledCode(
        computer=fixture_localhost,
        filepath_executable="/bin/true",
        default_calc_job_plugin="nanotech_empa.bader",
    ).store()


def test_bader_disabled_preserves_ot_input(scf_process):
    process = scf_process(
        run_diag_scf=orm.Bool(True), overlap_matrix=orm.Str("remote_only")
    )
    parameters = {"FORCE_EVAL": {"DFT": {"MGRID": {"CUTOFF": 300}}}}
    process.update_ot_input_dict(parameters)
    dft = parameters["FORCE_EVAL"]["DFT"]
    assert dft["MGRID"]["CUTOFF"] == 300
    assert "E_DENSITY_CUBE" not in dft.get("PRINT", {})
    assert not process.should_run_bader()
    assert process.should_run_diag_scf()


def test_bader_rejects_diag_scf(bader_code):
    message = Cp2kScfWorkChain._validate_inputs(
        {
            "overlap_matrix": orm.Str("none"),
            "run_diag_scf": orm.Bool(True),
            "bader_code": bader_code,
        },
        None,
    )
    assert "bader_code" in message


def test_bader_keeps_ot_only_density_settings(scf_process, bader_code):
    process = scf_process(
        bader_code=bader_code,
        overlap_matrix=orm.Str("remote_only"),
    )
    parameters = {"FORCE_EVAL": {"DFT": {"MGRID": {"CUTOFF": 300}}}}
    process.update_ot_input_dict(parameters)
    dft = parameters["FORCE_EVAL"]["DFT"]
    assert dft["MGRID"]["CUTOFF"] == 1200
    assert dft["PRINT"]["E_DENSITY_CUBE"] == {
        "STRIDE": "1 1 1",
        "EACH": {"QS_SCF": "0", "GEO_OPT": "0"},
        "ADD_LAST": "NUMERIC",
    }
    assert "AO_MATRICES" in dft["PRINT"]
    assert process.should_run_bader()
    assert not process.should_run_diag_scf()


@pytest.mark.parametrize("protocol", ["standard", "low_accuracy", "debug"])
def test_bader_forces_full_grid_density_from_protocol(
    scf_process, bader_code, protocol
):
    process = scf_process(bader_code=bader_code)
    parameters = cp2k_utils.load_protocol("scf_ot_protocol.yml", protocol)
    protocol_cube = copy.deepcopy(
        parameters["FORCE_EVAL"]["DFT"]["PRINT"]["E_DENSITY_CUBE"]
    )

    process.update_ot_input_dict(parameters)

    cube = parameters["FORCE_EVAL"]["DFT"]["PRINT"]["E_DENSITY_CUBE"]
    assert cube == {**protocol_cube, "STRIDE": "1 1 1"}


@pytest.mark.parametrize(
    ("scf_cutoff", "bader_cutoff", "expected"),
    [(300, 1200.0, 1200.0), (300, 900.0, 900.0), (1600, 1200.0, 1600)],
)
def test_bader_cutoff_is_a_lower_bound(
    scf_process, bader_code, scf_cutoff, bader_cutoff, expected
):
    process = scf_process(bader_code=bader_code, bader_cutoff=orm.Float(bader_cutoff))
    parameters = {"FORCE_EVAL": {"DFT": {"MGRID": {"CUTOFF": scf_cutoff}}}}
    process.update_ot_input_dict(parameters)
    assert parameters["FORCE_EVAL"]["DFT"]["MGRID"]["CUTOFF"] == expected


def _finished_calc(exit_status):
    node = orm.CalcJobNode()
    node.set_process_state(engine.ProcessState.FINISHED)
    node.set_exit_status(exit_status)
    node.set_exit_message("test failure" if exit_status else "")
    return node.store()


def test_bader_not_submitted_after_failed_ot_scf(scf_process, bader_code, monkeypatch):
    process = scf_process(bader_code=bader_code)
    process.ctx.ot_scf = _finished_calc(1)
    monkeypatch.setattr(
        process, "submit", lambda *_, **__: pytest.fail("Bader must not be submitted")
    )
    assert process.run_bader() == process.exit_codes.ERROR_TERMINATION


@pytest.mark.parametrize("failed_step", ["ot_scf", "bader"])
def test_bader_finalize_propagates_failures(scf_process, bader_code, failed_step):
    process = scf_process(bader_code=bader_code)
    for label in ("ot_scf", "bader"):
        process.ctx[label] = _finished_calc(1 if label == failed_step else 0)
    assert process.finalize() == process.exit_codes.ERROR_TERMINATION
    assert not process.outputs


@pytest.mark.parametrize("same_computer", [True, False])
def test_bader_submission_stages_density_and_retrieves_results(
    fixture_localhost, aiida_computer_local, bader_code, same_computer
):
    parent_computer = (
        fixture_localhost if same_computer else aiida_computer_local(label="parent")
    )
    parent = orm.RemoteData(computer=parent_computer, remote_path="/tmp/cp2k").store()
    process = (
        get_manager()
        .get_runner()
        .instantiate_process(
            BaderCalculation,
            code=bader_code,
            parent_calc_folder=parent,
            metadata={"options": {"resources": {"num_machines": 1}}},
        )
    )
    try:
        with SandboxFolder() as folder:
            calcinfo = process.prepare_for_submission(folder)
        assert not process.inputs.metadata.options.withmpi
        assert calcinfo.codes_info[0].cmdline_params == PINNED_BADER_OPTIONS + [
            "parent_calc_folder/aiida-ELECTRON_DENSITY-1_0.cube"
        ]
        assert calcinfo.retrieve_list == ["ACF.dat", "AVF.dat", "BCF.dat"]
        transfer = [(parent_computer.uuid, "/tmp/cp2k", "parent_calc_folder/")]
        assert calcinfo.remote_symlink_list == (transfer if same_computer else [])
        assert calcinfo.remote_copy_list == ([] if same_computer else transfer)
    finally:
        process.close()


def test_bader_additional_retrieve_list_extends(fixture_localhost, bader_code):
    process = (
        get_manager()
        .get_runner()
        .instantiate_process(
            BaderCalculation,
            code=bader_code,
            parent_calc_folder=orm.RemoteData(
                computer=fixture_localhost, remote_path="/tmp/cp2k"
            ).store(),
            settings=orm.Dict(dict={"additional_retrieve_list": ["aiida.out"]}),
            metadata={"options": {"resources": {"num_machines": 1}}},
        )
    )
    try:
        with SandboxFolder() as folder:
            calcinfo = process.prepare_for_submission(folder)
        assert calcinfo.retrieve_list == ["ACF.dat", "AVF.dat", "BCF.dat", "aiida.out"]
    finally:
        process.close()


def test_bader_custom_charge_density_filename(fixture_localhost, bader_code):
    process = (
        get_manager()
        .get_runner()
        .instantiate_process(
            BaderCalculation,
            code=bader_code,
            parent_calc_folder=orm.RemoteData(
                computer=fixture_localhost, remote_path="/tmp/cp2k"
            ).store(),
            charge_density_filename=orm.Str("density.cube"),
            metadata={"options": {"resources": {"num_machines": 1}}},
        )
    )
    try:
        with SandboxFolder() as folder:
            calcinfo = process.prepare_for_submission(folder)
        assert calcinfo.codes_info[0].cmdline_params == PINNED_BADER_OPTIONS + [
            "parent_calc_folder/density.cube"
        ]
    finally:
        process.close()


def test_bader_unknown_settings_are_rejected(fixture_localhost, bader_code):
    process = (
        get_manager()
        .get_runner()
        .instantiate_process(
            BaderCalculation,
            code=bader_code,
            parent_calc_folder=orm.RemoteData(
                computer=fixture_localhost, remote_path="/tmp/cp2k"
            ).store(),
            settings=orm.Dict(dict={"unexpected": True}),
            metadata={"options": {"resources": {"num_machines": 1}}},
        )
    )
    try:
        with SandboxFolder() as folder:
            with pytest.raises(common.InputValidationError, match="unexpected"):
                process.prepare_for_submission(folder)
    finally:
        process.close()


def _bader_node(computer, retrieved_files):
    node = orm.CalcJobNode(
        computer=computer,
        process_type="aiida.calculations:nanotech_empa.bader",
    )
    node.set_option("resources", {"num_machines": 1, "num_mpiprocs_per_machine": 1})
    node.store()

    retrieved = orm.FolderData()
    for name in retrieved_files:
        retrieved.base.repository.put_object_from_filelike(io.BytesIO(b""), name)
    retrieved.base.links.add_incoming(node, LinkType.CREATE, "retrieved")
    retrieved.store()
    return node


def test_bader_parser_outputs_present(fixture_localhost):
    node = _bader_node(fixture_localhost, ["ACF.dat", "AVF.dat", "BCF.dat"])

    assert BaderParser(node).parse() is None


@pytest.mark.parametrize("missing", ["ACF.dat", "AVF.dat", "BCF.dat"])
def test_bader_parser_output_missing(fixture_localhost, missing):
    files = [name for name in ("ACF.dat", "AVF.dat", "BCF.dat") if name != missing]
    node = _bader_node(fixture_localhost, files)

    exit_code = BaderParser(node).parse()
    assert exit_code.status == 300
