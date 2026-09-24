import io

from aiida import orm, plugins
from aiida.common.links import LinkType

SparseOverlapParser = plugins.ParserFactory("nanotech_empa.sparse_overlap")


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
