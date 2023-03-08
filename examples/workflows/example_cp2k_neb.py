import os
from ase import Atoms

from aiida.engine import run_get_node
from aiida.orm import Int, List, Str, Node, StructureData, load_code
from aiida.plugins import WorkflowFactory

Cp2kNebWorkChain = WorkflowFactory("nanotech_empa.cp2k.neb")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def _example_cp2k_neb(cp2k_code, uks, restart_uuid):

    builder = Cp2kNebWorkChain.get_builder()

    builder.metadata.label = "CP2K_NEB"
    builder.metadata.description = "a NEB calculation"
    builder.code = cp2k_code
    builder.options = {
        "max_wallclock_seconds": 600,
        "resources": {
            "num_machines": 3,
            "num_mpiprocs_per_machine": 1,
            "num_cores_per_mpiproc": 1,
        },
    }

    # define structures

    # check if test structure is already in database
    qb = QueryBuilder()
    qb.append(
        Node,
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
            ini = StructureData(
                ase=Atoms(
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
            rep = StructureData(
                ase=Atoms(
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
            fin = StructureData(
                ase=Atoms(
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

        builder.structure = load_node(uuids[0])
        replicas = {}
        for i in range(1, 3):
            name = "replica_%s" % str(i).zfill(3)
            replicas[name] = load_node(uuids[i])
        builder.replicas = replicas

    else:
        builder.structure = load_node(restart_uuid).inputs.structure
        builder.restart_from = Str(restart_uuid)

    dft_params = {
        "protocol": "debug",
        "cutoff": 300,
    }

    if uks:
        magnetization_per_site = [0, 1, -1, 0]
        dft_params = {
            "protocol": "debug",
            "uks": uks,
            "magnetization_per_site": magnetization_per_site,
            "charge": 0,
            "periodic": "NONE",
            "vdw": False,
            "multiplicity": 1,
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

    builder.dft_params = Dict(dict=dft_params)
    builder.sys_params = Dict(dict=sys_params)
    builder.neb_params = Dict(dict=neb_params)

    _, calc_node = run_get_node(builder)

    assert calc_node.is_finished_ok

    return calc_node.uuid


def example_cp2k_neb_rks(cp2k_code):
    _example_cp2k_neb(cp2k_code, False, None)


def example_cp2k_neb_uks(cp2k_code):
    _example_cp2k_neb(cp2k_code, True, None)


if __name__ == "__main__":
    print("####  RKS")
    uuid1 = _example_cp2k_neb(load_code("cp2k@localhost"), False, None)

    print("####  UKS")
    uuid2 = _example_cp2k_neb(load_code("cp2k@localhost"), True, None)

    print("### restarting from ", uuid2)
    uuid3 = _example_cp2k_neb(load_code("cp2k@localhost"), True, uuid2)
