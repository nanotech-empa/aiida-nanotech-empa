"""Report real child stderr even when a parser-less child exits successfully."""

import io

import pytest
from aiida import orm
from aiida.common.links import LinkType

from examples.workflows._diagnostics import assert_finished_ok


def test_failed_parent_includes_successful_child_stderr():
    parent = orm.WorkChainNode().store()
    parent.set_process_state("finished")
    parent.set_exit_status(390)
    child = orm.CalcJobNode()
    child.set_option("scheduler_stderr", "_scheduler-stderr.txt")
    child.set_option("scheduler_stdout", "_scheduler-stdout.txt")
    child.base.links.add_incoming(
        parent, link_type=LinkType.CALL_CALC, link_label="stm"
    )
    child.store()
    child.set_process_state("finished")
    child.set_exit_status(0)
    retrieved = orm.FolderData()
    retrieved.base.repository.put_object_from_filelike(
        io.BytesIO(b"ImportError: missing MPI library\n"), "_scheduler-stderr.txt"
    )
    retrieved.base.repository.put_object_from_filelike(
        io.BytesIO(b"SPM started\n"), "_scheduler-stdout.txt"
    )
    retrieved.base.links.add_incoming(
        child, link_type=LinkType.CREATE, link_label="retrieved"
    )
    retrieved.store()
    with pytest.raises(AssertionError) as error:
        assert_finished_ok(parent.uuid)
    assert "missing MPI library" in str(error.value)
    assert "SPM started" in str(error.value)
    assert str(child.uuid) in str(error.value)
    assert "exit_status=0" in str(error.value)


def test_successful_parent_does_not_read_child_streams(monkeypatch):
    parent = orm.WorkChainNode().store()
    parent.set_process_state("finished")
    parent.set_exit_status(0)
    monkeypatch.setattr(
        orm.CalcJobNode,
        "get_scheduler_stderr",
        lambda *args: pytest.fail("Unexpected stream read"),
    )
    assert_finished_ok(parent.uuid)
