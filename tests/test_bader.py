"""Regression tests for the optional OT-only Bader branch."""

import copy
import io
import shutil

import ase
import numpy as np
import pytest
from aiida import common, engine, orm, plugins
from aiida.common.folders import SandboxFolder
from aiida.common.links import LinkType
from aiida.manage import get_manager

BaderCalculation = plugins.CalculationFactory("nanotech_empa.bader")
BaderParser = plugins.ParserFactory("nanotech_empa.bader")
Cp2kScfWorkChain = plugins.WorkflowFactory("nanotech_empa.cp2k.scf")


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
    original = copy.deepcopy(parameters)
    process.update_ot_input_dict(parameters)
    assert parameters == original
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


@pytest.mark.parametrize("failed_step", ["ot_scf", "bader"])
def test_bader_finalize_propagates_failures(scf_process, bader_code, failed_step):
    process = scf_process(bader_code=bader_code)
    for label in ("ot_scf", "bader"):
        node = orm.CalcJobNode()
        node.set_process_state(engine.ProcessState.FINISHED)
        node.set_exit_status(1 if label == failed_step else 0)
        node.set_exit_message("test failure" if label == failed_step else "")
        process.ctx[label] = node.store()
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
        assert calcinfo.codes_info[0].cmdline_params == [
            "parent_calc_folder/aiida-ELECTRON_DENSITY-1_0.cube"
        ]
        assert calcinfo.retrieve_list == ["ACF.dat", "AVF.dat", "BCF.dat"]
        transfer = [(parent_computer.uuid, "/tmp/cp2k", "parent_calc_folder/")]
        assert calcinfo.remote_symlink_list == (transfer if same_computer else [])
        assert calcinfo.remote_copy_list == ([] if same_computer else transfer)
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


def test_cp2k_scf_bader_retrieves_charges(cp2k_code, local_code_factory):
    if not shutil.which("bader"):
        pytest.skip("Bader executable not available")
    builder = Cp2kScfWorkChain.get_builder()
    builder.cp2k_code = cp2k_code
    builder.bader_code = local_code_factory("nanotech_empa.bader", "bader")
    builder.structure = orm.StructureData(
        ase=ase.Atoms(
            "H2", positions=[[4, 4, 3.63], [4, 4, 4.37]], cell=[8, 8, 8], pbc=True
        )
    )
    builder.protocol = orm.Str("debug")
    builder.dft_params = orm.Dict(dict={"periodic": "XYZ", "cutoff": 150})
    builder.options = orm.Dict(
        dict={
            "max_wallclock_seconds": 600,
            "resources": {"num_machines": 1, "num_mpiprocs_per_machine": 1},
        }
    )
    _, node = engine.run_get_node(builder)
    assert node.is_finished_ok
    assert len(node.called) == 2  # OT SCF and Bader, without diagonalization.
    retrieved = node.outputs.bader_retrieved
    assert {"ACF.dat", "AVF.dat", "BCF.dat"} <= set(
        retrieved.base.repository.list_object_names()
    )
    with retrieved.base.repository.open("ACF.dat") as handle:
        charges = np.loadtxt(handle, skiprows=2, max_rows=2)
    assert np.isfinite(charges).all()
    np.testing.assert_allclose(charges[:, 4].sum(), 2.0, atol=0.02)
    bader = next(
        child for child in node.called if child.process_label == "BaderCalculation"
    )
    assert bader.inputs.parent_calc_folder.uuid == node.outputs.remote_folder.uuid
    assert bader.outputs.retrieved.uuid == retrieved.uuid
