import os
import subprocess
import time

import click
import numpy as np
from aiida import engine, orm, plugins

# from aiida.manage.manager import get_manager
Cp2kReftrajWorkChain = plugins.WorkflowFactory("nanotech_empa.cp2k.reftraj")
StructureData = orm.DataFactory("core.structure")
TrajectoryData = orm.DataFactory("core.array.trajectory")


def _example_cp2k_reftraj(cp2k_code, num_batches=2, restart=False, to_be_killed=False):
    os.path.dirname(os.path.realpath(__file__))

    # Structure.
    # structure = StructureData(ase=ase.io.read(os.path.join(thisdir, ".", "h2.xyz")))

    # check if input trajectory already in database otherwise create it
    qb = orm.QueryBuilder()
    qb.append(TrajectoryData, filters={"label": "H2_trajectory"})
    if qb.count() == 0:
        steps = 60
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
        trajectory.label = "H2_trajectory"
        trajectory.store()
        print("stored trajectory ", trajectory.pk)
    else:
        trajectory = qb.first()[0]

    builder = Cp2kReftrajWorkChain.get_builder()

    builder.metadata.label = "CP2K_RefTraj"
    builder.metadata.description = "test description"
    builder.code = cp2k_code
    if to_be_killed:
        builder.options = {
            "max_wallclock_seconds": 300,
            "mpirun_extra_params": [
                "timeout",
                "1",
            ],  # Kill the calculation after 1 second to test the restart failure.
            "resources": {
                "num_machines": 1,
                "num_mpiprocs_per_machine": 1,
                "num_cores_per_mpiproc": 1,
            },
        }
    else:
        builder.options = {
            "max_wallclock_seconds": 300,
            "resources": {
                "num_machines": 1,
                "num_mpiprocs_per_machine": 1,
                "num_cores_per_mpiproc": 1,
            },
        }

    # builder.structure = structure
    builder.trajectory = trajectory
    if restart:
        builder.restart = orm.Bool(True)
    builder.num_batches = orm.Int(num_batches)
    builder.protocol = orm.Str("debug")

    dft_params = {
        "uks": True,
        "magnetization_per_site": [0, 1],
        "charge": 0,
        "periodic": "NONE",
        "vdw": False,
        "multiplicity": 1,
        "cutoff": 300,
    }

    sys_params = {}
    builder.dft_params = orm.Dict(dft_params)
    builder.sys_params = orm.Dict(sys_params)

    # _, calc_node = engine.run_get_node(builder)
    # we use submit to be able to kill the workchain
    calc_node = engine.submit(builder)
    return calc_node.pk


def example_cp2k_reftraj(cp2k_code):
    _example_cp2k_reftraj(cp2k_code)
    # _example_cp2k_replicachain(cp2k_code, pk1)


@click.command("cli")
@click.argument("cp2k_code", default="cp2k@localhost")
@click.option("-n", "--n-nodes", default=1)
@click.option("-c", "--n-cores-per-node", default=1)
def run_all(cp2k_code, n_nodes, n_cores_per_node):
    # running reftraj with 1 batch to have a reference trajectory
    count = 0
    maxcount = 8
    print("#### UKS one batch")
    pk1 = _example_cp2k_reftraj(cp2k_code=orm.load_code(cp2k_code), num_batches=1)
    wait = True
    while wait:
        print("waiting for first workchain to finish")
        wc = orm.load_node(pk1)
        wait = not wc.is_finished
        count += 1
        if count > maxcount:
            print("timeout")
            break
        time.sleep(10)

    # running from scratch reftraj with 4 batches the ccalcjob will not have enough time to complete and we will kill the workchain as son as the second calculation is finished
    print("#### UKS four batches")
    pk2 = _example_cp2k_reftraj(
        cp2k_code=orm.load_code(cp2k_code),
        num_batches=4,
        restart=False,
        to_be_killed=True,
    )
    can_stop = False
    finished = 0
    count = 0
    maxcount = 8
    while not can_stop:
        count += 1
        if count > maxcount:
            print("timeout")
            break

        print("waiting for second workchain to finish")
        wc = orm.load_node(pk2)
        for calc in wc.called_descendants:
            if calc.process_label == "Cp2kCalculation" and calc.is_finished:
                finished += 1
                if finished > 1:
                    can_stop = True
        if can_stop:
            command = ["verdi", "process", "kill", str(pk2)]
            # Execute the command using subprocess
            try:
                result = subprocess.run(
                    command, check=True, capture_output=True, text=True
                )
                print(f"Successfully killed workchain with PK {pk2}.")
                print(result.stdout)
            except subprocess.CalledProcessError as e:
                print(f"Error while killing workchain with PK {pk2}:")
                print(e.stderr)
            print(f"waiting for {pk2} to be killed ")
            time.sleep(10)
        else:
            time.sleep(5)

    # restart from the previuously killed workchain
    print("#### UKS three batches restart")
    pk3 = _example_cp2k_reftraj(
        cp2k_code=orm.load_code(cp2k_code), num_batches=3, restart=True
    )
    wait = True
    count = 0
    maxcount = 8
    while wait:
        print("waiting for third workchain to finish")
        wc = orm.load_node(pk3)
        wait = not wc.is_finished
        count += 1
        if count > maxcount:
            print("timeout")
            break
        time.sleep(10)
    traj1 = orm.load_node(pk1).outputs.output_trajectory
    traj3 = orm.load_node(pk3).outputs.output_trajectory
    print("#### DONE ####")
    assert np.allclose(
        traj1.get_array("cells"),
        traj3.get_array("cells"),
        rtol=1e-07,
        atol=1e-08,
        equal_nan=False,
    )
    print("arrays match")


if __name__ == "__main__":
    run_all()
