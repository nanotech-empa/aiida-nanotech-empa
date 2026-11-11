"""STM result guard with synthetic archives and real FolderData/CalcJob links."""

import copy
import io
import zipfile
import zlib
from types import SimpleNamespace as NS

import numpy as np
import pytest
from aiida import orm
from aiida.common.links import LinkType
from ase import Atoms
from plumpy import ProcessState

from aiida_nanotech_empa.workflows.cp2k.stm_workchain import Cp2kStmWorkChain

pytest_plugins = ["aiida.tools.pytest_fixtures"]


def workflow(version):
    return Cp2kStmWorkChain


@pytest.fixture(autouse=True)
def isolated(aiida_profile_clean):
    assert aiida_profile_clean.is_test_profile


def archive(maps=None, **overrides):
    if maps is None:
        maps = np.zeros((2, 3, 4, 5), dtype=np.float32)
    payload = {
        "stm_general_info": {
            "energies": np.arange(3),
            "x_arr": np.arange(5),
            "y_arr": np.arange(6),
        },
        "stm_series_info": [{"type": "const-height stm", "height": 3.0, "fwhm": 0.1}]
        * 2,
        "stm_series_data": maps,
    }
    payload.update(overrides)
    buf = io.BytesIO()
    np.savez(buf, **payload)
    return buf.getvalue()


def actual_writer(current, mode):
    data = np.zeros((2, 3, 4, 5)) if mode == "zero" else np.ones((2, 3, 4, 5))
    if mode == "missing_crossing":
        data[1] = -2.0
    info = {
        "energies": np.array([-3.1, -2.6, -2.1]),
        "x_arr": np.arange(4),
        "y_arr": np.arange(5),
    }
    if current:
        info.update(cell_vectors=np.diag([4.0, 5.0, 6.0]), dv_vectors=np.eye(3))
    series = [
        {"type": "const-height stm", "height": 3.0, "fwhm": 0.1},
        {
            "type": "const-isovalue stm"
            if mode == "missing_crossing"
            else "const-height sts",
            "isovalue": 1e-7,
            "fwhm": 0.1,
        },
    ]
    buf = io.BytesIO()
    np.savez_compressed(
        buf, stm_general_info=info, stm_series_info=series, stm_series_data=data
    )
    return buf.getvalue()


def helper(payload, state=ProcessState.FINISHED, exit_status=0, retrieved=True):
    n = orm.CalcJobNode(process_type="aiida.calculations:nanotech_empa.stm").store()
    n.set_process_state(state)
    n.set_exit_status(exit_status)
    if retrieved:
        folder = orm.FolderData()
        if payload is not None:
            folder.base.repository.put_object_from_filelike(
                io.BytesIO(payload), "stm.npz"
            )
        folder.base.links.add_incoming(
            n, link_type=LinkType.CREATE, link_label="retrieved"
        )
        folder.store()
    n.seal()
    return n


def finalize(node, version="candidate"):
    cls = workflow(version)
    source = orm.StructureData(
        ase=Atoms("H", positions=[[6, 6, 6]], cell=[12] * 3, pbc=True)
    ).store()
    before = copy.deepcopy(source.base.attributes.all)
    extras_before = copy.deepcopy(source.base.extras.all)
    reports = []
    outputs = {}
    dft = orm.Dict({"energy": -1.0})
    h = NS(
        ctx=NS(stm=node, diag_scf=NS(outputs=NS(output_parameters=dft))),
        inputs=NS(structure=source),
        node=NS(uuid="private-synthetic-finalization"),
        exit_codes=cls.exit_codes,
        report=reports.append,
        out=lambda k, v: outputs.update({k: v}),
    )
    value = cls.finalize(h)
    assert source.base.attributes.all == before
    if value is not None:
        assert source.base.extras.all == extras_before
    return value, reports, outputs, source.base.extras.all


@pytest.mark.parametrize(
    "state,status",
    [
        (ProcessState.FINISHED, 1),
        (ProcessState.FINISHED, 501),
        (ProcessState.EXCEPTED, None),
        (ProcessState.KILLED, None),
    ],
)
def test_failed_helper_with_valid_stale_file_cannot_pass(state, status):
    n = helper(archive(), state, status)
    value, reports, outputs, extras = finalize(n)
    assert value.status == 390 and "did not finish successfully" in reports[-1]
    assert not outputs and "surfaces" not in extras


@pytest.mark.parametrize("retrieved", [True, False])
def test_missing_result_returns_explicit_error(retrieved):
    value, reports, outputs, extras = finalize(helper(None, retrieved=retrieved))
    assert value.status == 390 and (
        "no stm.npz" in reports[-1] or "no retrieved" in reports[-1]
    )
    assert not outputs and "surfaces" not in extras


def invalid(kind):
    if kind == "zero_bytes":
        return b""
    if kind == "text":
        return b"helper failed, no results"
    if kind == "empty_npz":
        buf = io.BytesIO()
        np.savez(buf)
        return buf.getvalue()
    if kind == "truncated_zip":
        return archive()[:-60]
    if kind == "missing_entry":
        buf = io.BytesIO()
        np.savez(buf, stm_series_data=np.ones((1, 2, 3, 4)))
        return buf.getvalue()
    if kind == "wrong_dimension":
        return archive(np.ones((2, 4, 5)))
    if kind == "object_maps":
        return archive(np.full((2, 3, 4, 5), None, dtype=object))
    if kind == "text_maps":
        return archive(np.full((2, 3, 4, 5), "x"))
    if kind.startswith("empty_axis"):
        shape = [2, 3, 4, 5]
        shape[int(kind[-1])] = 0
        return archive(np.zeros(shape))
    if kind == "damaged_metadata":
        raw = bytearray(archive())
        marker = b"const-height stm"
        index = raw.index(marker)
        raw[index] ^= 1
        return bytes(raw)
    raise AssertionError(kind)


@pytest.mark.parametrize(
    "kind",
    [
        "zero_bytes",
        "text",
        "empty_npz",
        "truncated_zip",
        "missing_entry",
        "wrong_dimension",
        "object_maps",
        "text_maps",
        "empty_axis0",
        "empty_axis1",
        "empty_axis2",
        "empty_axis3",
        "damaged_metadata",
    ],
)
def test_unusable_archive_returns_error_without_success_links(kind):
    value, reports, outputs, extras = finalize(helper(invalid(kind)))
    assert value.status == 390 and "unreadable or empty" in reports[-1]
    assert not outputs and "surfaces" not in extras


@pytest.mark.parametrize(
    "current", [False, True], ids=["legacy_metadata", "current_vectors"]
)
@pytest.mark.parametrize("mode", ["normal", "zero", "missing_crossing"])
def test_legacy_and_current_formats_and_zero_maps_remain_valid(current, mode):
    payload = actual_writer(current, mode)
    n = helper(payload)
    value, reports, outputs, extras = finalize(n)
    assert value is None and "Work chain is finished" in reports[-1]
    assert outputs["dft_output_parameters"].get_dict() == {"energy": -1.0}
    assert extras["surfaces"] == ["private-synthetic-finalization"]
    with n.outputs.retrieved.base.repository.open("stm.npz", "rb") as f:
        assert f.read() == payload


def test_real_compressed_payload_corruption_returns_390():
    buffer = io.BytesIO()
    np.savez_compressed(buffer, stm_series_data=np.arange(240).reshape(2, 3, 4, 10))
    raw = bytearray(buffer.getvalue())
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        info = archive.getinfo("stm_series_data.npy")
    # Read local-header lengths: numpy ZIP64 can use local extra fields even
    # when central-directory ZipInfo.extra is empty.
    import struct

    name_length, extra_length = struct.unpack_from("<HH", raw, info.header_offset + 26)
    offset = info.header_offset + 30 + name_length + extra_length
    raw[offset] ^= 255
    corrupted = bytes(raw)
    with zipfile.ZipFile(io.BytesIO(corrupted)) as archive:
        with pytest.raises(zlib.error):
            archive.testzip()
    result, reports, outputs, extras = finalize(helper(corrupted))
    assert result.status == 390 and "unreadable or empty" in reports[-1]
    assert not outputs and "surfaces" not in extras
