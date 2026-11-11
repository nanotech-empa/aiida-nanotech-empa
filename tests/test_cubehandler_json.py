"""Pure input-generation checks for optional render JSON companions."""

import json
from types import SimpleNamespace as NS

import pytest
import yaml
from aiida import common
from aiida.common.folders import SandboxFolder

from aiida_nanotech_empa.plugins.cubehandler import (
    CubeHandlerCalculation,
    json_companion_files,
)


def test_render_json_files_are_staged_with_recorded_parameters():
    parameters = {
        "steps": [
            {
                "command": "render",
                "args": ["source/full.cube"],
                "options": {"lighting": "lighting.json", "slice_plane": "plane.json"},
            }
        ],
        "json_files": {
            "lighting.json": {"ambient": 0.4},
            "plane.json": {"periodic": True, "offset": 0.25},
        },
    }
    context = NS(
        uuid="synthetic-only",
        inputs=NS(
            parameters=NS(get_dict=lambda: parameters),
            code=NS(uuid="synthetic-code"),
            parent_folders={},
        ),
        _DEFAULT_INPUT_FILE="aiida.inp",
    )
    with SandboxFolder() as folder:
        info = CubeHandlerCalculation.prepare_for_submission(context, folder)
        for name, value in parameters["json_files"].items():
            with folder.open(name) as handle:
                assert json.load(handle) == value
        with folder.open("aiida.inp") as handle:
            assert yaml.safe_load(handle) == parameters
        assert info.codes_info[0].cmdline_params == ["run", "aiida.inp"]
        assert info.retrieve_list == ["out_cubes"]


def test_existing_parameters_need_no_companion_files():
    assert json_companion_files({"steps": []}) == {}


@pytest.mark.parametrize(
    "values",
    [
        [],
        {"../outside.json": {}},
        {"/outside.json": {}},
        {"aiida.inp": {}},
        {".hidden.json": {}},
        {"bad.json": {"x": float("nan")}},
        {"bad.json": {"x": object()}},
    ],
)
def test_invalid_companions_are_rejected_before_writing(values):
    with pytest.raises(common.InputValidationError):
        json_companion_files({"json_files": values})
