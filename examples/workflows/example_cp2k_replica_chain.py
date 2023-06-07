import pathlib

import ase.io
import click
from aiida import engine, orm, plugins

Cp2kReplicaWorkChain = plugins.WorkflowFactory("nanotech_empa.cp2k.replica")

DATA_DIR = pathlib.Path(__file__).parent.absolute()
GEO_FILE = "c2h4.xyz"


def _example_cp2k_replicachain(
    cp2k_code, targets, restart_uuid, n_nodes, n_cores_per_node
):
    # check if test geometry is already in database
    qb = orm.QueryBuilder()
    qb.append(orm.Node, filters={"label": {"in": [GEO_FILE]}})
    structure = None
    for node_tuple in qb.iterall():
        node = node_tuple[0]
        structure = node
    if structure is not None:
        print("found existing structure: ", structure.pk)
    else:
        structure = orm.StructureData(ase=ase.io.read(DATA_DIR / GEO_FILE))
        structure.label = GEO_FILE
        structure.store()
        print("created new structure: ", structure.pk)

    builder = Cp2kReplicaWorkChain.get_builder()
    if restart_uuid is not None:
        builder.restart_from = orm.Str(restart_uuid)

    builder.metadata.label = "CP2K_Replica"
    builder.metadata.description = "test description"
    builder.code = cp2k_code
    builder.options = {
        "max_wallclock_seconds": 600,
        "resources": {
            "num_machines": n_nodes,
            "num_mpiprocs_per_machine": n_cores_per_node,
            "num_cores_per_mpiproc": 1,
        },
    }

    builder.structure = orm.StructureData(ase=ase.io.read(DATA_DIR / GEO_FILE))
    builder.protocol = orm.Str("debug")

    dft_params = {
        "periodic": "NONE",
        "vdw": False,
        "cutoff": 300,
    }

    sys_params = {}
    sys_params[
        "constraints"
    ] = "fixed 1 , collective 1 [ev/angstrom^2] 40 [angstrom] 1.33 , collective 2 [ev/angstrom^2] 40 [angstrom] 1.09, collective 3 [ev/angstrom^2] 40 [angstrom] 1.87"
    sys_params[
        "colvars"
    ] = "distance atoms 1 2 , distance atoms 1 3, distance atoms 5 6"
    sys_params["colvars_targets"] = targets
    sys_params["colvars_increments"] = [0.06, 0.05, 0.0]
    builder.dft_params = orm.Dict(dft_params)
    builder.sys_params = orm.Dict(sys_params)

    _, calc_node = engine.run_get_node(builder)

    assert calc_node.is_finished_ok
    return calc_node.pk


def example_cp2k_replicachain_rks(cp2k_code):
    pk1 = _example_cp2k_replicachain(cp2k_code, None)
    _example_cp2k_replicachain(cp2k_code, pk1)


@click.command("cli")
@click.argument("cp2k_code", default="cp2k@localhost")
@click.option("-n", "--n-nodes", default=1)
@click.option("-c", "--n-cores-per-node", default=1)
def run_all(cp2k_code, n_nodes, n_cores_per_node):
    print("#### RKS")
    uuid = _example_cp2k_replicachain(
        cp2k_code=orm.load_code(cp2k_code),
        targets=[1.40, 1.21, 1.87],
        restart_uuid=None,
        n_nodes=n_nodes,
        n_cores_per_node=n_cores_per_node,
    )
    print(f"#### RKS continuation from uuid ({uuid})")
    _example_cp2k_replicachain(
        cp2k_code=orm.load_code(cp2k_code),
        targets=[1.47, 1.27, 1.87],
        restart_uuid=uuid,
        n_nodes=n_nodes,
        n_cores_per_node=n_cores_per_node,
    )


if __name__ == "__main__":
    run_all()
