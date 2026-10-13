"""Parser failures must return supported CP2K exit codes, not raise AttributeError."""

import io

import pytest
from aiida import orm, plugins
from aiida.common.links import LinkType

from aiida_nanotech_empa.parsers.cp2k_gw_parser import Cp2kGwParser
from aiida_nanotech_empa.parsers.cp2k_neb_parser import Cp2kNebParser


@pytest.fixture
def retrieved_calc():
    node = orm.CalcJobNode(process_type="aiida.calculations:cp2k").store()
    retrieved = orm.FolderData()
    retrieved.base.links.add_incoming(
        node, link_type=LinkType.CREATE, link_label="retrieved"
    )
    yield node, retrieved


def test_neb_missing_stdout_returns_output_missing(retrieved_calc):
    node, retrieved = retrieved_calc
    retrieved.store()
    result = Cp2kNebParser(node)._parse_stdout()
    assert result == plugins.CalculationFactory("cp2k").exit_codes.ERROR_OUTPUT_MISSING


@pytest.mark.parametrize(
    ("parser_class", "method", "file_attribute"),
    [
        (Cp2kNebParser, "_parse_stdout", "_DEFAULT_OUTPUT_FILE"),
        (Cp2kNebParser, "_parse_trajectory", "_DEFAULT_RESTART_FILE_NAME"),
        (Cp2kGwParser, "_parse_stdout", "_DEFAULT_OUTPUT_FILE"),
    ],
)
def test_unreadable_output_returns_output_read(
    retrieved_calc, monkeypatch, parser_class, method, file_attribute
):
    node, retrieved = retrieved_calc
    name = getattr(node.process_class, file_attribute)
    retrieved.base.repository.put_object_from_filelike(
        io.BytesIO(b"present but unreadable"), name
    )
    retrieved.store()

    def failed_read(repository, path, *args, **kwargs):
        raise OSError("test repository read failure")

    monkeypatch.setattr(
        type(retrieved.base.repository), "get_object_content", failed_read
    )
    result = getattr(parser_class(node), method)()
    assert result == plugins.CalculationFactory("cp2k").exit_codes.ERROR_OUTPUT_READ
