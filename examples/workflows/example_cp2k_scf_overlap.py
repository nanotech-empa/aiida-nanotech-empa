import ase.io
import click
import numpy as np
from aiida import engine, orm, plugins

try:
    from examples.workflows._paths import script_dir
except ModuleNotFoundError:
    from _paths import script_dir

Cp2kScfWorkChain = plugins.WorkflowFactory("nanotech_empa.cp2k.scf")

DATA_DIR = script_dir(__file__)
GEO_FILE = "ch4.xyz"


def _check_sparse_overlap(retrieved):
    assert "sparse_overlap.npz" in retrieved.base.repository.list_object_names()

    with retrieved.base.repository.open("sparse_overlap.npz", "rb") as handle:
        with np.load(handle) as sparse_overlap:
            shape = tuple(sparse_overlap["shape"])
            elements = set(sparse_overlap["element"].astype(str))
            assert shape[0] == shape[1]
            assert shape[0] == sparse_overlap["basis_index"].size
            assert sparse_overlap["data"].size >= shape[0]
            assert {"C", "H"} <= elements


def _example_cp2k_scf(
    cp2k_code, sparse_overlap_code, run_diag_scf, n_nodes=1, n_cores_per_node=1
):
    builder = Cp2kScfWorkChain.get_builder()

    builder.metadata.label = "Cp2kScfWorkChain"
    builder.cp2k_code = cp2k_code
    builder.structure = orm.StructureData(ase=ase.io.read(DATA_DIR / GEO_FILE))
    builder.protocol = orm.Str("debug")
    builder.dft_params = orm.Dict({"periodic": "XYZ", "cutoff": 150})
    builder.options = orm.Dict(
        {
            "max_wallclock_seconds": 600,
            "resources": {
                "num_machines": n_nodes,
                "num_mpiprocs_per_machine": n_cores_per_node,
                "num_cores_per_mpiproc": 1,
            },
        }
    )
    builder.run_diag_scf = orm.Bool(run_diag_scf)
    builder.overlap_matrix = orm.Str("remote_and_sparse_retrieved")
    builder.sparse_overlap_code = sparse_overlap_code
    builder.overlap_threshold = orm.Float(1.0e-10)

    _, calc_node = engine.run_get_node(builder)

    assert calc_node.is_finished_ok
    # OT SCF, optional diagonalization SCF, sparse overlap post-processing.
    assert len(calc_node.called) == (3 if run_diag_scf else 2)
    _check_sparse_overlap(calc_node.outputs.sparse_overlap_retrieved)


def example_cp2k_scf_ot_sparse_overlap(cp2k_code, sparse_overlap_code):
    _example_cp2k_scf(cp2k_code, sparse_overlap_code, run_diag_scf=False)


def example_cp2k_scf_diag_sparse_overlap(cp2k_code, sparse_overlap_code):
    _example_cp2k_scf(cp2k_code, sparse_overlap_code, run_diag_scf=True)


@click.command("cli")
@click.argument("cp2k_code", default="cp2k@localhost")
@click.argument("sparse_overlap_code", default="sparse_overlap@localhost")
@click.option("-n", "--n-nodes", default=1)
@click.option("-c", "--n-cores-per-node", default=1)
def run_all(cp2k_code, sparse_overlap_code, n_nodes, n_cores_per_node):
    for run_diag_scf in (False, True):
        print(f"#### sparse AO overlap, run_diag_scf={run_diag_scf}")
        _example_cp2k_scf(
            orm.load_code(cp2k_code),
            orm.load_code(sparse_overlap_code),
            run_diag_scf=run_diag_scf,
            n_nodes=n_nodes,
            n_cores_per_node=n_cores_per_node,
        )


if __name__ == "__main__":
    run_all()
