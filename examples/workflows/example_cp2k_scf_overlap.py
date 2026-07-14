import numpy as np
import click
from ase import Atoms
from aiida import engine, orm, plugins

Cp2kScfWorkChain = plugins.WorkflowFactory("nanotech_empa.cp2k.scf")


def methane_structure():
    center = np.array([4.0, 4.0, 4.0])
    delta = 1.09 / np.sqrt(3.0)
    offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [delta, delta, delta],
            [delta, -delta, -delta],
            [-delta, delta, -delta],
            [-delta, -delta, delta],
        ]
    )
    atoms = Atoms(
        "CH4",
        positions=center + offsets,
        cell=[8.0, 8.0, 8.0],
        pbc=True,
    )
    return orm.StructureData(ase=atoms)


def run_example(cp2k_code, sparse_overlap_code, n_nodes=1, n_cores_per_node=1):
    builder = Cp2kScfWorkChain.get_builder()
    builder.metadata.label = "Cp2kScfWorkChain CH4 sparse overlap example"
    builder.cp2k_code = cp2k_code
    builder.sparse_overlap_code = sparse_overlap_code
    builder.structure = methane_structure()
    builder.protocol = orm.Str("debug")
    builder.dft_params = orm.Dict(dict={"periodic": "XYZ", "cutoff": 150})
    builder.options = orm.Dict(
        dict={
            "max_wallclock_seconds": 600,
            "resources": {
                "num_machines": n_nodes,
                "num_mpiprocs_per_machine": n_cores_per_node,
                "num_cores_per_mpiproc": 1,
            },
        }
    )
    builder.retrieve_sparse_overlap = orm.Bool(True)
    builder.overlap_threshold = orm.Float(1.0e-10)

    _, node = engine.run_get_node(builder)

    print(f"WorkChain PK: {node.pk}")
    print(f"Finished OK: {node.is_finished_ok}")
    if not node.is_finished_ok:
        print(f"Exit status: {node.exit_status}")
        print(f"Exit message: {node.exit_message}")
        return node

    retrieved = node.outputs.sparse_overlap_retrieved
    names = retrieved.base.repository.list_object_names()
    print("Sparse overlap retrieved files:", names)
    assert "sparse_overlap.npz" in names
    return node


@click.command("cli")
@click.argument("cp2k_code", default="cp2k@localhost")
@click.argument("sparse_overlap_code", default="sparse_overlap@localhost")
@click.option("-n", "--n-nodes", default=1, show_default=True)
@click.option("-c", "--n-cores-per-node", default=1, show_default=True)
def main(cp2k_code, sparse_overlap_code, n_nodes, n_cores_per_node):
    run_example(
        cp2k_code=orm.load_code(cp2k_code),
        sparse_overlap_code=orm.load_code(sparse_overlap_code),
        n_nodes=n_nodes,
        n_cores_per_node=n_cores_per_node,
    )


if __name__ == "__main__":
    main()
