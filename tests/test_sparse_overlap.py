import io

from aiida import orm, plugins
from aiida.common.folders import SandboxFolder
from aiida.common.links import LinkType
from aiida.engine.utils import instantiate_process
from aiida.manage import get_manager

SparseOverlapCalculation = plugins.CalculationFactory("nanotech_empa.sparse_overlap")
SparseOverlapParser = plugins.ParserFactory("nanotech_empa.sparse_overlap")


def test_sparse_overlap_additional_retrieve_list_extends(aiida_localhost):
    code = orm.InstalledCode(
        computer=aiida_localhost,
        filepath_executable="/bin/true",
        default_calc_job_plugin="nanotech_empa.sparse_overlap",
    ).store()
    process = instantiate_process(
        get_manager().get_runner(),
        SparseOverlapCalculation,
        code=code,
        parent_calc_folder=orm.RemoteData(
            computer=aiida_localhost, remote_path="/tmp/parent"
        ),
        settings=orm.Dict({"additional_retrieve_list": ["aiida.out"]}),
        metadata={"options": {"resources": {"num_machines": 1}}},
    )

    with SandboxFolder() as folder:
        calcinfo = process.prepare_for_submission(folder)

    assert calcinfo.retrieve_list == ["sparse_overlap.npz", "aiida.out"]


def _sparse_overlap_node(computer, retrieved_files):
    node = orm.CalcJobNode(
        computer=computer,
        process_type="aiida.calculations:nanotech_empa.sparse_overlap",
    )
    node.set_option("resources", {"num_machines": 1, "num_mpiprocs_per_machine": 1})
    output_filename = orm.Str("sparse_overlap.npz").store()
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


def test_sparse_overlap_parser_output_present(aiida_localhost):
    node = _sparse_overlap_node(aiida_localhost, ["sparse_overlap.npz"])

    assert SparseOverlapParser(node).parse() is None


def test_sparse_overlap_parser_output_missing(aiida_localhost):
    node = _sparse_overlap_node(aiida_localhost, [])

    exit_code = SparseOverlapParser(node).parse()
    assert exit_code.status == 300
