import io

import pytest
from aiida import orm, plugins
from aiida.common.folders import SandboxFolder
from aiida.common.links import LinkType
from aiida.engine.utils import instantiate_process
from aiida.manage import get_manager

from aiida_nanotech_empa.plugins import unfolding

Cp2kUnfoldingCalculation = plugins.CalculationFactory("nanotech_empa.cp2k_unfolding")
Cp2kUnfoldingParser = plugins.ParserFactory("nanotech_empa.cp2k_unfolding")


def test_unfolding_additional_retrieve_list_extends(aiida_localhost):
    code = orm.InstalledCode(
        computer=aiida_localhost,
        filepath_executable="/bin/true",
        default_calc_job_plugin="nanotech_empa.cp2k_unfolding",
    ).store()
    process = instantiate_process(
        get_manager().get_runner(),
        Cp2kUnfoldingCalculation,
        code=code,
        parent_calc_folder=orm.RemoteData(
            computer=aiida_localhost, remote_path="/tmp/parent"
        ),
        primitive_vectors=orm.Str("1 0 0; 0 1 0"),
        settings=orm.Dict({"additional_retrieve_list": ["aiida.out"]}),
        metadata={"options": {"resources": {"num_machines": 1}}},
    )

    with SandboxFolder() as folder:
        calcinfo = process.prepare_for_submission(folder)

    assert calcinfo.retrieve_list == ["unfolding_bands.npz", "aiida.out"]


def _unfolding_node(computer, retrieved_files):
    node = orm.CalcJobNode(
        computer=computer,
        process_type="aiida.calculations:nanotech_empa.cp2k_unfolding",
    )
    node.set_option("resources", {"num_machines": 1, "num_mpiprocs_per_machine": 1})
    output_filename = orm.Str("unfolding_bands.npz").store()
    node.base.links.add_incoming(
        output_filename, LinkType.INPUT_CALC, "output_filename"
    )
    node.store()

    retrieved = orm.FolderData()
    for name in retrieved_files:
        retrieved.base.repository.put_object_from_filelike(io.BytesIO(b""), name)
    retrieved.base.links.add_incoming(node, LinkType.CREATE, "retrieved")
    retrieved.store()
    return node


def test_unfolding_parser_requires_output_file(aiida_localhost):
    present = _unfolding_node(aiida_localhost, ["unfolding_bands.npz"])
    assert Cp2kUnfoldingParser(present).parse() is None

    missing = _unfolding_node(aiida_localhost, [])
    assert Cp2kUnfoldingParser(missing).parse().status == 300


@pytest.mark.parametrize(
    ("lattice_type", "valid"), [("hexagonal", True), ("hexagnal", False)]
)
def test_validate_lattice_type(lattice_type, valid):
    message = unfolding.validate_lattice_type(orm.Str(lattice_type), None)

    if valid:
        assert message is None
    else:
        assert "hexagonal" in message


@pytest.mark.parametrize(
    ("primitive_vectors", "valid"),
    [
        ("2.46 0 0; -1.23 2.13 0", True),
        ("2.46, 0, 0\n-1.23, 2.13, 0", True),
        ("4.0 0 0", True),
        ("", False),
        ("2.46 0; -1.23 2.13", False),
        ("a b c", False),
        ("1 0 0; 0 1 0; 0 0 1", False),
    ],
)
def test_validate_primitive_vectors(primitive_vectors, valid):
    message = unfolding.validate_primitive_vectors(orm.Str(primitive_vectors), None)

    if valid:
        assert message is None
    else:
        assert "three numbers" in message


@pytest.mark.parametrize(
    ("window", "error"),
    [
        ({}, None),
        ({"emin": -2.0, "emax": 2.0}, None),
        ({"emin": -2.0}, "set together"),
        ({"emin": 2.0, "emax": -2.0}, "lower than"),
    ],
)
def test_validate_energy_window(window, error):
    inputs = {key: orm.Float(bound) for key, bound in window.items()}

    message = unfolding.validate_energy_window(inputs, "emin", "emax")

    if error is None:
        assert message is None
    else:
        assert error in message
