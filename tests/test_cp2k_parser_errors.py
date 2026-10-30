"""Missing, unreadable or incomplete CP2K outputs return Cp2kCalculation exit codes."""

import io

import pytest
from aiida import orm
from aiida.common.links import LinkType
from aiida.orm.nodes.repository import NodeRepository

from aiida_nanotech_empa.parsers.cp2k_gw_parser import Cp2kGwParser
from aiida_nanotech_empa.parsers.cp2k_neb_parser import Cp2kNebParser


def _cp2k_node(file_attribute=None):
    node = orm.CalcJobNode(process_type="aiida.calculations:cp2k").store()
    retrieved = orm.FolderData()
    if file_attribute:
        name = getattr(node.process_class, file_attribute)
        retrieved.base.repository.put_object_from_filelike(io.BytesIO(b"x"), name)
    retrieved.base.links.add_incoming(
        node, link_type=LinkType.CREATE, link_label="retrieved"
    )
    retrieved.store()
    return node


@pytest.mark.parametrize(
    ("parser_class", "method", "file_attribute"),
    [
        (Cp2kNebParser, "_parse_stdout", "_DEFAULT_OUTPUT_FILE"),
        (Cp2kNebParser, "_parse_trajectory", "_DEFAULT_RESTART_FILE_NAME"),
        (Cp2kGwParser, "_parse_stdout", "_DEFAULT_OUTPUT_FILE"),
    ],
)
@pytest.mark.parametrize(
    ("present", "expected"),
    [(False, "ERROR_OUTPUT_MISSING"), (True, "ERROR_OUTPUT_READ")],
)
def test_unavailable_output(
    monkeypatch, parser_class, method, file_attribute, present, expected
):
    node = _cp2k_node(file_attribute if present else None)

    def failed_read(*args, **kwargs):
        raise OSError

    monkeypatch.setattr(NodeRepository, "get_object_content", failed_read)
    result = getattr(parser_class(node), method)()
    assert result == node.process_class.exit_codes[expected]


def test_gw_incomplete_output():
    node = _cp2k_node("_DEFAULT_OUTPUT_FILE")
    result = Cp2kGwParser(node)._parse_stdout()
    assert result == node.process_class.exit_codes.ERROR_OUTPUT_INCOMPLETE
