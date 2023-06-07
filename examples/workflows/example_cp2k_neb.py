import pathlib

import ase
import click
from aiida import engine, orm, plugins

Cp2kNebWorkChain = plugins.WorkflowFactory("nanotech_empa.cp2k.neb")

DATA_DIR = pathlib.Path(__file__).parent.absolute()


def _example_cp2k_neb(cp2k_code, uks, restart_uuid, n_nodes, n_cores_per_node):
    builder = Cp2kNebWorkChain.get_builder()

    builder.metadata.label = "CP2K_NEB"
    builder.metadata.description = "NEB calculation."
    builder.code = cp2k_code
    builder.options = {
        "max_wallclock_seconds": 600,
        "resources": {
            "num_machines": n_nodes,
            "num_mpiprocs_per_machine": n_cores_per_node,
            "num_cores_per_mpiproc": 1,
        },
    }

    # define structures

    # Check if test structure is already in the database.
    qb = orm.QueryBuilder()
    qb.append(
        orm.Node,
        filters={
            "label": {
                "in": [
                    "auto_test_neb_ini_structure",
                    "auto_test_neb_rep_structure",
                    "auto_test_neb_fin_structure",
                ]
            }
        },
    )
    if restart_uuid is None:
        available_structures = {}
        uuids = []
        for node_tuple in qb.iterall():
            node = node_tuple[0]
            available_structures[node.label] = node.uuid

        if "auto_test_neb_ini_structure" in available_structures:
            uuids = [available_structures["auto_test_neb_ini_structure"]]
            print("found ini: ", available_structures["auto_test_neb_ini_structure"])
        else:
            ini = orm.StructureData(
                ase=ase.Atoms(
                    "HCCH",
                    positions=[
                        [1, 2, 2],
                        [2.07, 2, 2],
                        [3.43, 2, 2],
                        [4.50, 2, 2],
                    ],
                    cell=[6.0, 4.0, 4.0],
                )
            )
            ini.label = "auto_test_neb_ini_structure"
            ini.store()
            uuids = [ini.uuid]
            print("created ini: ", ini.uuid)
        if "auto_test_neb_rep_structure" in available_structures:
            uuids.append(available_structures["auto_test_neb_rep_structure"])
            print("found rep: ", available_structures["auto_test_neb_rep_structure"])
        else:
            rep = orm.StructureData(
                ase=ase.Atoms(
                    "HCCH",
                    positions=[
                        [1, 2, 2],
                        [2.07, 2, 2],
                        [3.43, 2, 2],
                        [4.60, 2, 2],
                    ],
                    cell=[6.0, 4.0, 4.0],
                )
            )
            rep.label = "auto_test_neb_rep_structure"
            rep.store()
            uuids.append(rep.uuid)
            print("created rep: ", rep.uuid)
        if "auto_test_neb_fin_structure" in available_structures:
            uuids.append(available_structures["auto_test_neb_fin_structure"])
            print("found fin: ", available_structures["auto_test_neb_fin_structure"])
        else:
            fin = orm.StructureData(
                ase=ase.Atoms(
                    "HCCH",
                    positions=[
                        [1, 2, 2],
                        [2.07, 2, 2],
                        [3.43, 2, 2],
                        [4.70, 2, 2],
                    ],
                    cell=[6.0, 4.0, 4.0],
                )
            )
            fin.label = "auto_test_neb_fin_structure"
            fin.store()
            uuids.append(fin.uuid)
            print("created fin: ", fin.uuid)

        builder.structure = orm.load_node(uuids[0])
        replicas = {}
        for i in range(1, 3):
            name = f"replica_{str(i).zfill(3)}"
            replicas[name] = orm.load_node(uuids[i])
        builder.replicas = replicas

    else:
        builder.structure = orm.load_node(restart_uuid).inputs.structure
        builder.restart_from = orm.Str(restart_uuid)

    builder.protocol = orm.Str("debug")

    if uks:
        magnetization_per_site = [0, 1, -1, 0]
        dft_params = {
            "uks": uks,
            "magnetization_per_site": magnetization_per_site,
            "charge": 0,
            "periodic": "NONE",
            "vdw": False,
            "multiplicity": 1,
            "cutoff": 300,
        }
    else:
        dft_params = {
            "cutoff": 300,
        }

    sys_params = {}
    sys_params[
        "constraints"
    ] = "fixed xyz 1 , collective 1 [ev/angstrom^2] 40 [angstrom] 1.36"
    sys_params["colvars"] = "distance atoms 2 3"
    neb_params = {
        "align_frames": ".TRUE.",
        "band_type": "CI-NEB",
        "k_spring": 0.1,
        "nproc_rep": 1,
        "number_of_replica": 3,
        "nsteps_it": 1,
        "optimize_end_points": ".TRUE.",
    }

    builder.dft_params = orm.Dict(dict=dft_params)
    builder.sys_params = orm.Dict(dict=sys_params)
    builder.neb_params = orm.Dict(dict=neb_params)

    _, calc_node = engine.run_get_node(builder)

    assert calc_node.is_finished_ok

    return calc_node.uuid


def example_cp2k_neb_rks(cp2k_code):
    _example_cp2k_neb(cp2k_code, False, None)


def example_cp2k_neb_uks(cp2k_code):
    _example_cp2k_neb(cp2k_code, True, None)


@click.command("cli")
@click.argument("cp2k_code", default="cp2k@localhost")
@click.option("-n", "--n-nodes", default=3)
@click.option("-c", "--n-cores-per-node", default=1)
def run_all(cp2k_code, n_nodes, n_cores_per_node):
    print("####  RKS")
    _example_cp2k_neb(
        orm.load_code(cp2k_code),
        uks=False,
        restart_uuid=None,
        n_nodes=n_nodes,
        n_cores_per_node=n_cores_per_node,
    )

    print("####  UKS")
    uuid2 = _example_cp2k_neb(
        orm.load_code(cp2k_code),
        uks=True,
        restart_uuid=None,
        n_nodes=n_nodes,
        n_cores_per_node=n_cores_per_node,
    )

    print("### restarting from ", uuid2)
    _example_cp2k_neb(
        orm.load_code(cp2k_code),
        uks=True,
        restart_uuid=uuid2,
        n_nodes=n_nodes,
        n_cores_per_node=n_cores_per_node,
    )


if __name__ == "__main__":
    run_all()
