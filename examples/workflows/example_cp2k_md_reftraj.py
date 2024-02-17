import os

import ase.io
import click
import numpy as np
from aiida import engine, orm, plugins

Cp2kReftrajWorkChain = plugins.WorkflowFactory("nanotech_empa.cp2k.reftraj")
StructureData = DataFactory("core.structure")
TrajectoryData = DataFactory("core.array.trajectory")


def _example_cp2k_reftraj(cp2k_code, num_batches=2):
    os.path.dirname(os.path.realpath(__file__))

    # Structure.
    # structure = StructureData(ase=ase.io.read(os.path.join(thisdir, ".", "h2.xyz")))

    # Trajectory.
    steps = 20
    positions = np.array(
        [
            [
                [2.528, 3.966, 3.75 + 0.0001 * i],
                [2.528, 3.966, 3],
            ]
            for i in range(steps)
        ]
    )
    cells = np.array(
        [[[5, 0, 0], [0, 5, 0], [0, 0, 5 + 0.0001 * i]] for i in range(steps)]
    )
    symbols = ["H", "H"]
    trajectory = TrajectoryData()
    trajectory.set_trajectory(symbols, positions, cells=cells)

    builder = Cp2kReftrajWorkChain.get_builder()
    # if restart_uuid is not None:
    #    builder.restart_from = orm.Str(restart_uuid)

    builder.metadata.label = "CP2K_RefTraj"
    builder.metadata.description = "test description"
    builder.code = cp2k_code
    builder.options = {
        "max_wallclock_seconds": 600,
        "resources": {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
            "num_cores_per_mpiproc": 1,
        },
    }

    # builder.structure = structure
    builder.trajectory = trajectory
    builder.num_batches = orm.Int(num_batches)
    builder.protocol = orm.Str("debug")

    dft_params = {
        "periodic": "NONE",
        "vdw": False,
        "cutoff": 200,
    }

    sys_params = {}
    builder.dft_params = orm.Dict(dft_params)
    builder.sys_params = orm.Dict(sys_params)

    _, calc_node = engine.run_get_node(builder)

    assert calc_node.is_finished_ok
    return calc_node.pk


def example_cp2k_reftraj(cp2k_code):
    _example_cp2k_reftraj(cp2k_code)
    # _example_cp2k_replicachain(cp2k_code, pk1)


@click.command("cli")
@click.argument("cp2k_code", default="cp2k@localhost")
@click.option("-n", "--n-nodes", default=1)
@click.option("-c", "--n-cores-per-node", default=1)
def run_all(cp2k_code, n_nodes, n_cores_per_node):
    print("#### RKS one batch")
    uuid1 = _example_cp2k_reftraj(cp2k_code=orm.load_code(cp2k_code), num_batches=1)
    print("#### RKS two batches")
    uuid2 = _example_cp2k_reftraj(cp2k_code=orm.load_code(cp2k_code), num_batches=2)
    traj1 = orm.load_node(uuid1).outputs.output_trajectory
    traj2 = orm.load_node(uuid2).outputs.output_trajectory
    assert np.allclose(
        traj1.get_array("cells"),
        traj2.get_array("cells"),
        rtol=1e-07,
        atol=1e-08,
        equal_nan=False,
    )
    print(f"arrays  match")


if __name__ == "__main__":
    run_all()
