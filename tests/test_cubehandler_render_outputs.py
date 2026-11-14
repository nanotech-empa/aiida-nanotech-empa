"""Synthetic repository/image checks, including a successful shell with no PNG."""

import io
import tomllib
from pathlib import Path

import pytest
from aiida import orm
from aiida.common.links import LinkType
from PIL import Image

from aiida_nanotech_empa.parsers.cubehandler_parser import CubeHandlerParser
from aiida_nanotech_empa.plugins.cubehandler import CubeHandlerCalculation
from aiida_nanotech_empa.utils.cubehandler_results import rendered_images


def image_bytes(image_format="PNG", size=(16, 12)):
    stream = io.BytesIO()
    Image.new("RGB", size, "#336699").save(stream, format=image_format)
    return stream.getvalue()


def parameters(**options):
    return {
        "steps": [
            {
                "command": "render",
                "options": {
                    "output": "out_cubes/render.png",
                    "format": "png",
                    "width": 16,
                    "height": 12,
                    **options,
                },
            }
        ]
    }


@pytest.fixture
def parser_for(aiida_profile_clean, aiida_localhost):
    assert aiida_profile_clean.is_test_profile

    def make(values, files=None):
        node = orm.CalcJobNode(
            process_type="aiida.calculations:nanotech_empa.cubehandler",
            computer=aiida_localhost,
        )
        node.base.links.add_incoming(
            orm.Dict(values).store(), LinkType.INPUT_CALC, "parameters"
        )
        node.store()
        if files is not None:
            retrieved = orm.FolderData()
            for name, data in files.items():
                retrieved.base.repository.put_object_from_filelike(
                    io.BytesIO(data), name
                )
            retrieved.store()
            retrieved.base.links.add_incoming(node, LinkType.CREATE, "retrieved")
        return CubeHandlerParser(node)

    return make


def test_parser_is_declared_and_default():
    declaration = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text()
    )
    assert (
        declaration["project"]["entry-points"]["aiida.parsers"][
            "nanotech_empa.cubehandler"
        ]
        == "aiida_nanotech_empa.parsers.cubehandler_parser:CubeHandlerParser"
    )
    assert (
        CubeHandlerCalculation.spec()
        .inputs["metadata"]["options"]["parser_name"]
        .default
        == "nanotech_empa.cubehandler"
    )


@pytest.mark.parametrize(
    "files",
    [
        None,
        {},
        {"out_cubes/render.png": b""},
        {"out_cubes/render.png": b"not an image"},
    ],
)
def test_no_or_corrupt_image_is_failure_even_if_scheduler_finished(parser_for, files):
    parser = parser_for(parameters(), files)
    assert parser.parse() == parser.exit_codes.ERROR_RENDER_OUTPUT


@pytest.mark.parametrize("data", [image_bytes(size=(15, 12)), image_bytes("JPEG")])
def test_size_and_format_must_match_request(parser_for, data):
    parser = parser_for(parameters(), {"out_cubes/render.png": data})
    assert parser.parse() == parser.exit_codes.ERROR_RENDER_OUTPUT


def test_valid_image_is_accepted_and_repository_is_unchanged(parser_for):
    parser = parser_for(parameters(), {"out_cubes/render.png": image_bytes()})
    before = parser.retrieved.base.repository.metadata
    assert parser.parse() is None
    assert parser.retrieved.base.repository.metadata == before
    assert rendered_images(parameters(), parser.retrieved.base.repository.open) == [
        {
            "filename": "out_cubes/render.png",
            "size": (16, 12),
            "format": "PNG",
        }
    ]


def test_multiple_views_require_every_image(parser_for):
    values = parameters()
    values["steps"] += parameters(output="out_cubes/second.png")["steps"]
    parser = parser_for(values, {"out_cubes/render.png": image_bytes()})
    assert parser.parse() == parser.exit_codes.ERROR_RENDER_OUTPUT


@pytest.mark.parametrize(
    "requested,stored",
    [
        ("png", "PNG"),
        ("jpg", "JPEG"),
        ("jpeg", "JPEG"),
        ("tif", "TIFF"),
        ("tiff", "TIFF"),
        ("bmp", "BMP"),
        ("pnm", "PPM"),
        ("ps", "EPS"),
    ],
)
def test_existing_cli_formats_are_preserved(parser_for, requested, stored):
    path = f"out_cubes/render.{requested}"
    parser = parser_for(
        parameters(output=path, format=requested), {path: image_bytes(stored)}
    )
    assert parser.parse() is None


@pytest.mark.parametrize(
    "output,filename",
    [(None, "density_render.png"), ("out_cubes/render", "out_cubes/render.png")],
)
def test_cli_default_and_extensionless_output_names(parser_for, output, filename):
    values = parameters(output=output)
    values["steps"][0]["args"] = ["full_cubes/density.cube"]
    parser = parser_for(values, {filename: image_bytes()})
    assert parser.parse() is None


@pytest.mark.parametrize("output", ["", "../outside.png", "/outside.png"])
def test_invalid_output_is_rejected_without_access(parser_for, output):
    parser = parser_for(parameters(output=output), {})
    assert parser.parse() == parser.exit_codes.ERROR_RENDER_OUTPUT


@pytest.mark.parametrize("command", ["sum", "shrink"])
def test_nonrender_steps_keep_existing_contract_without_retrieval(parser_for, command):
    assert parser_for({"steps": [{"command": command}]}).parse() is None
