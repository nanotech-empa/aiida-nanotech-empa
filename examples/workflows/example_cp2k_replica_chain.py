import os
from ase import Atoms
from ase.io import read
from aiida.orm import StructureData, Int, List, Str
from aiida.orm import load_code
from aiida.engine import run_get_node
from aiida.plugins import WorkflowFactory

Cp2kReplicaWorkChain = WorkflowFactory("nanotech_empa.cp2k.replica")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
GEO_FILE = "c2h4.xyz"


def _example_cp2k_replicachain(cp2k_code, targets, restart_uuid):

    # check if test geometry is already in database
    qb = QueryBuilder()
    qb.append(Node, filters={"label": {"in": [GEO_FILE]}})
    structure = None
    for node_tuple in qb.iterall():
        node = node_tuple[0]
        structure = node
    if structure is not None:
        print("found existing structure: ", structure.pk)
    else:
        structure = StructureData(ase=read(os.path.join(DATA_DIR, GEO_FILE)))
        structure.label = GEO_FILE
        structure.store()
        print("created new structure: ", structure.pk)

    builder = Cp2kReplicaWorkChain.get_builder()
    if restart_uuid is not None:
        builder.restart_from = Str(restart_uuid)

    builder.metadata.label = "CP2K_Replica"
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

    builder.structure = StructureData(ase=read(os.path.join(DATA_DIR, GEO_FILE)))

    dft_params = {
        "protocol": "debug",
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
    builder.dft_params = Dict(dft_params)
    builder.sys_params = Dict(sys_params)

    _, calc_node = run_get_node(builder)

    assert calc_node.is_finished_ok

    # replicachain_out_dict = dict(calc_node.outputs.output_parameters)
    # print()
    # for k in replicachain_out_dict:
    #    print("  {}: {}".format(k, replicachain_out_dict[k]))
    return calc_node.pk


def example_cp2k_replicachain_rks(cp2k_code):
    pk1 = _example_cp2k_replicachain(cp2k_code, None)
    pk2 = _example_cp2k_replicachain(cp2k_code, pk1)


# def example_cp2k_replicachain_rks_continuation(cp2k_code):
#    _example_cp2k_replicachain(cp2k_code,1):
#    _example_cp2k_slabopt(cp2k_code, 1)


if __name__ == "__main__":
    print("#### RKS")
    pk1 = _example_cp2k_replicachain(
        load_code("cp2k@localhost"), [1.40, 1.21, 1.87], None
    )
    print(f"#### RKS continuation from pk {pk1}")
    pk2 = _example_cp2k_replicachain(
        load_code("cp2k@localhost"), [1.47, 1.27, 1.87], pk1
    )

#    print("#### UKS")
#    _example_cp2k_replicachain(load_code("cp2k@localhost"), 1)
