"""Exercise the generated runner against the released irrep Espresso parser."""

import xml.etree.ElementTree as ET

import h5py
import numpy as np
import pytest
from irrep.bandstructure import ParserEspresso as BandStructureParser
from irrep.parsers.espresso import ParserEspresso

from aiida_nanotech_empa.plugins.qe_banduppy import _runner_script


def espresso_files(tmp_path, mode="none", uppercase=False):
    """Write small QE-format files with independently known complex coefficients."""
    prefix = tmp_path / "aiida"
    saved = tmp_path / "aiida.save"
    saved.mkdir()
    spinor = mode == "spinor"
    polarized = mode in ("up", "dw")
    nbands = 3 if mode == "dw" else 4
    root = ET.Element("espresso")
    inputs = ET.SubElement(root, "input")
    basis = ET.SubElement(inputs, "basis")
    ET.SubElement(basis, "ecutwfc").text = "30"
    output = ET.SubElement(root, "output")
    reciprocal = ET.SubElement(ET.SubElement(output, "basis_set"), "reciprocal_lattice")
    for index, row in enumerate(np.eye(3), start=1):
        ET.SubElement(reciprocal, f"b{index}").text = " ".join(map(str, row))
    bands = ET.SubElement(output, "band_structure")
    ET.SubElement(bands, "noncolin").text = str(spinor).lower()
    if polarized:
        ET.SubElement(bands, "nbnd_up").text = "4"
        ET.SubElement(bands, "nbnd_dw").text = "3"
    else:
        ET.SubElement(bands, "nbnd").text = str(nbands)
    ET.SubElement(bands, "fermi_energy").text = "0.1"
    kpoint = ET.SubElement(bands, "ks_energies")
    ET.SubElement(kpoint, "k_point").text = "0 0 0"
    ET.SubElement(kpoint, "npw").text = "3"
    eigenvalues = np.arange(7 if polarized else nbands) * 0.2 - 1.3
    ET.SubElement(kpoint, "eigenvalues").text = " ".join(map(str, eigenvalues))
    ET.ElementTree(root).write(saved / "data-file-schema.xml")

    columns = 3 * (2 if spinor else 1)
    numbers = np.arange(nbands * columns).reshape(nbands, columns)
    coefficients = (0.1 + numbers) * np.exp(1j * (0.2 + numbers * 0.3))
    interleaved = np.empty((nbands, 2 * columns))
    interleaved[:, 0::2] = coefficients.real
    interleaved[:, 1::2] = coefficients.imag
    filename = f"wfc{mode if polarized else ''}1"
    if uppercase:
        filename = filename.upper()
    filename = saved / (filename + ".hdf5")
    with h5py.File(filename, "w") as handle:
        handle.attrs.update(
            gamma_only=False,
            igwx=3,
            ik=1,
            ispin=2 if mode == "dw" else 1,
            nbnd=nbands,
            ngw=3,
            npol=2 if spinor else 1,
            scale_factor=1.0,
            xk=np.zeros(3),
        )
        miller = handle.create_dataset(
            "MillerIndices", data=[[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        )
        for index, row in enumerate(np.eye(3), start=1):
            miller.attrs[f"bg{index}"] = row
        handle.create_dataset("evc", data=interleaved)
    return str(prefix), filename, nbands


@pytest.fixture
def runner(monkeypatch):
    # The generated helper patches only its own subprocess. Restore the real
    # released parser methods between tests, including failures.
    monkeypatch.setattr(ParserEspresso, "parse_header", ParserEspresso.parse_header)
    monkeypatch.setattr(ParserEspresso, "parse_kpoint", ParserEspresso.parse_kpoint)
    namespace = {"__name__": "reader_contract_test"}
    # Execute only our trusted generated helper, without its __main__ entry.
    exec(compile(_runner_script(), "run_banduppy_qe.py", "exec"), namespace)  # noqa: S102
    return namespace


@pytest.mark.parametrize("mode", ["none", "up", "dw", "spinor"])
@pytest.mark.parametrize("start,stop", [(None, None), (1, 3), (-2, None)])
def test_sliced_reader_matches_official_complex_coefficients_and_energies(
    tmp_path,
    runner,
    mode,
    start,
    stop,
):
    prefix, _, nbands = espresso_files(tmp_path, mode)
    channel = mode if mode in ("up", "dw") else None
    original = ParserEspresso(prefix)
    original.parse_header(spin_channel=channel)
    expected = original.parse_kpoint(0)
    assert BandStructureParser is ParserEspresso
    if start is None and stop is None:
        assert runner["_patch_espresso_hdf5_band_slice"](start, stop) is False
    else:
        assert runner["_patch_espresso_hdf5_band_slice"](start, stop) is True
    parser = ParserEspresso(prefix)
    header = parser.parse_header(spin_channel=channel)
    # Actual current irrep.BandStructure calls these keyword arguments.
    actual = parser.parse_kpoint(0, verbosity=0, getE=True, getWF=True)
    selected = slice(start, stop)
    assert header[-1] == len(range(nbands)[selected])
    np.testing.assert_array_equal(actual[0], expected[0][selected])
    np.testing.assert_array_equal(actual[1], expected[1][selected])
    np.testing.assert_array_equal(actual[2], expected[2])
    np.testing.assert_array_equal(actual[3], expected[3])


@pytest.mark.parametrize(
    "get_energy,get_wavefunction", [(True, False), (False, True), (False, False)]
)
def test_reader_flags_do_not_load_unrequested_wavefunction_bands(
    tmp_path,
    runner,
    monkeypatch,
    get_energy,
    get_wavefunction,
):
    prefix, _, _ = espresso_files(tmp_path)
    original = ParserEspresso(prefix)
    original.parse_header()
    expected = original.parse_kpoint(0)
    reads = []
    getitem = h5py.Dataset.__getitem__

    def observe_read(dataset, selection):
        if dataset.name == "/evc":
            reads.append(selection)
        return getitem(dataset, selection)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", observe_read)
    runner["_patch_espresso_hdf5_band_slice"](1, 3)
    parser = ParserEspresso(prefix)
    parser.parse_header()
    wf, energies, kg, kpoint = parser.parse_kpoint(
        0, getE=get_energy, getWF=get_wavefunction
    )
    if get_wavefunction:
        np.testing.assert_array_equal(wf, expected[0][1:3])
        assert reads == [(slice(1, 3), slice(None))]
    else:
        assert wf is None
        assert reads == []
    if get_energy:
        np.testing.assert_array_equal(energies, expected[1][1:3])
    else:
        assert energies is None
    np.testing.assert_array_equal(kg, expected[2])
    np.testing.assert_array_equal(kpoint, expected[3])


def test_uppercase_filename_and_bounded_hdf5_read(tmp_path, runner, monkeypatch):
    prefix, _, _ = espresso_files(tmp_path, uppercase=True)
    reads = []
    getitem = h5py.Dataset.__getitem__

    def observe_read(dataset, selection):
        if dataset.name == "/evc":
            reads.append(selection)
        return getitem(dataset, selection)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", observe_read)
    runner["_patch_espresso_hdf5_band_slice"](1, 3)
    parser = ParserEspresso(prefix)
    parser.parse_header()
    parser.parse_kpoint(0, getE=True, getWF=True)
    assert reads == [(slice(1, 3), slice(None))]


@pytest.mark.parametrize("start,stop", [(4, 5), (2, 1), (-5, 3)])
def test_invalid_band_slice_is_rejected(tmp_path, runner, start, stop):
    prefix, _, _ = espresso_files(tmp_path)
    runner["_patch_espresso_hdf5_band_slice"](start, stop)
    with pytest.raises(RuntimeError, match="Invalid band slice"):
        ParserEspresso(prefix).parse_header()


@pytest.mark.parametrize("corruption", ["index", "coordinate"])
def test_reader_preserves_modern_kpoint_identity_guards(tmp_path, runner, corruption):
    prefix, filename, _ = espresso_files(tmp_path)
    with h5py.File(filename, "r+") as handle:
        if corruption == "index":
            handle.attrs["ik"] = 2
        else:
            handle.attrs["xk"] = [0.25, 0, 0]
    runner["_patch_espresso_hdf5_band_slice"](1, 3)
    parser = ParserEspresso(prefix)
    parser.parse_header()
    with pytest.raises(RuntimeError, match="k-point"):
        parser.parse_kpoint(0, getE=True, getWF=True)
