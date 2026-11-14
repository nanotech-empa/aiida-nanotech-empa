"""Exercise fragment builders and the released CP2K handler; no native jobs."""

import copy
import io

import pytest
import test_fragment_serial
from aiida import orm, plugins
from aiida.common.links import LinkType
from aiida.engine import ProcessState
from aiida.engine.utils import instantiate_process
from test_fragment_serial import run_case

source_inputs = test_fragment_serial.source_inputs

Base = plugins.WorkflowFactory("cp2k.base")
Calculation = plugins.CalculationFactory("cp2k")


@pytest.mark.parametrize("serial", [False, True])
def test_all_fragment_builders_enable_restart_with_default_budget(
    source_inputs, serial
):
    _, node, _, payloads, _ = run_case(source_inputs, serial=serial)
    assert node.is_finished_ok
    for row in payloads[:-1]:
        assert row["handler_overrides"] == {
            "restart_incomplete_calculation": {"enabled": True}
        }
        assert row["max_iterations"] == 5
        assert row["on_unhandled_failure"] == "pause"
        assert row["pause_on_max_iterations"] is True


def failed_calculation(computer, structure, directory, kind, *, evidence=True):
    """A typed terminal child with stored retrieved/restart provenance only."""
    node = orm.CalcJobNode(process_type="aiida.calculations:cp2k", computer=computer)
    node.base.attributes.set("output_filename", "aiida.out")
    node.store()
    node.set_process_state(ProcessState.FINISHED)
    node.set_exit_status(Calculation.exit_codes.ERROR_OUT_OF_WALLTIME.status)
    content = "SCF run terminated - exceeded requested execution time: 300 seconds.\n"
    if evidence:
        content += "Total energy: -17.0\n"
        if kind == "GEO_OPT":
            content += "Max. gradient = 0.001\n"
    retrieved = orm.FolderData()
    retrieved.base.repository.put_object_from_filelike(
        io.StringIO(content), "aiida.out"
    )
    retrieved.store()
    retrieved.base.links.add_incoming(node, LinkType.CREATE, "retrieved")
    directory.mkdir()
    (directory / "aiida-RESTART.wfn").write_bytes(b"synthetic-test-only")
    remote = orm.RemoteData(computer=computer, remote_path=str(directory)).store()
    remote.base.links.add_incoming(node, LinkType.CREATE, "remote_folder")
    result = structure.clone().store()
    result.base.links.add_incoming(node, LinkType.CREATE, "output_structure")
    return node, remote, result


@pytest.mark.parametrize("kind", ["ENERGY", "GEO_OPT"])
@pytest.mark.parametrize("attempts", [1, 5])
def test_actual_handler_recovers_walltime_and_respects_budget(
    source_inputs, aiida_manager, tmp_path, monkeypatch, kind, attempts
):
    inputs = dict(source_inputs, max_iterations=orm.Int(attempts))
    _, _, _, payloads, _ = run_case(inputs, serial=True)
    row = next(
        r for r in payloads[:-1] if r["parameters"]["GLOBAL"]["RUN_TYPE"] == kind
    )
    params = orm.Dict(copy.deepcopy(row["parameters"]))
    before = copy.deepcopy(params.get_dict())
    process = instantiate_process(
        aiida_manager.get_runner(),
        Base,
        cp2k={
            "code": inputs["code"],
            "structure": inputs["structure"],
            "parameters": params,
            "metadata": {"options": row["options"]},
        },
        handler_overrides=orm.Dict(row["handler_overrides"]),
        max_iterations=orm.Int(row["max_iterations"]),
        on_unhandled_failure=orm.Str(row["on_unhandled_failure"]),
        pause_on_max_iterations=orm.Bool(row["pause_on_max_iterations"]),
    )
    try:
        process.setup()
        child, remote, result = failed_calculation(
            inputs["code"].computer, inputs["structure"], tmp_path / "completed", kind
        )
        process.ctx.children = [child]
        process.ctx.iteration = 1
        pauses = []
        monkeypatch.setattr(process, "pause", pauses.append)
        report = process.inspect_process()
        assert report is None or report.status == 0
        assert process.ctx.inputs.parent_calc_folder.uuid == remote.uuid
        repaired = process.ctx.inputs.parameters.get_dict()
        assert (
            repaired["FORCE_EVAL"]["DFT"]["SCF"]["EPS_SCF"]
            == before["FORCE_EVAL"]["DFT"]["SCF"]["EPS_SCF"]
        )
        assert "IGNORE_CONVERGENCE_FAILURE" not in repaired["FORCE_EVAL"]["DFT"]["SCF"]
        assert (
            "EXT_RESTART" in repaired
            if kind == "GEO_OPT"
            else "EXT_RESTART" not in repaired
        )
        assert repaired["GLOBAL"] == before["GLOBAL"]
        assert repaired["FORCE_EVAL"]["DFT"]["XC"] == before["FORCE_EVAL"]["DFT"]["XC"]
        assert (
            process.ctx.inputs["metadata"]["options"]["resources"]
            == row["options"]["resources"]
        )
        process.overwrite_input_structure()
        assert process.ctx.inputs.structure.uuid == result.uuid
        assert params.get_dict() == before
        assert bool(pauses) is (attempts == 1)
        assert process.ctx.iteration == (0 if attempts == 1 else 1)
        assert not process.ctx.is_finished
    finally:
        process.close()


def test_actual_handler_rejects_child_without_restart_evidence(
    source_inputs, aiida_manager, tmp_path
):
    _, _, _, payloads, _ = run_case(source_inputs, serial=True)
    row = payloads[0]
    process = instantiate_process(
        aiida_manager.get_runner(),
        Base,
        cp2k={
            "code": source_inputs["code"],
            "structure": source_inputs["structure"],
            "parameters": orm.Dict(row["parameters"]),
            "metadata": {"options": row["options"]},
        },
        handler_overrides=orm.Dict(row["handler_overrides"]),
        max_iterations=orm.Int(5),
        on_unhandled_failure=orm.Str("abort"),
    )
    try:
        process.setup()
        child, _, _ = failed_calculation(
            source_inputs["code"].computer,
            source_inputs["structure"],
            tmp_path / "not-restartable",
            "ENERGY",
            evidence=False,
        )
        process.ctx.children = [child]
        process.ctx.iteration = 1
        report = process.inspect_process()
        assert report == process.exit_codes.NO_RESTART_DATA
        assert "parent_calc_folder" not in process.ctx.inputs
    finally:
        process.close()
