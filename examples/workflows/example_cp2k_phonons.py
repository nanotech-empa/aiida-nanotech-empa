import os
import ase.io
from ase import Atoms

from aiida.engine import run_get_node
from aiida.orm import StructureData, load_code
from aiida.plugins import WorkflowFactory

Cp2kPhononsWorkChain = WorkflowFactory("nanotech_empa.cp2k.phonons")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
GEO_FILE = "c2h2.xyz"


def _example_cp2k_phonons(cp2k_code, uks):
    # check test geometry is already in database
    qb = QueryBuilder()
    qb.append(Node, filters={"label": {"in": [GEO_FILE]}})
    structure = None
    for node_tuple in qb.iterall():
        node = node_tuple[0]
        structure = node
    if structure is not None:
        print("found existing structure: ", structure.pk)
    else:
        structure = StructureData(ase=ase.io.read(os.path.join(DATA_DIR, GEO_FILE)))
        structure.label = GEO_FILE
        structure.store()
        print("created new structure: ", structure.pk)

    builder = Cp2kPhononsWorkChain.get_builder()

    builder.metadata.label = "CP2K_Phonons"
    builder.metadata.description = "test phonons c2h2"
    builder.code = cp2k_code
    builder.options = {
        "max_wallclock_seconds": 600,
        "resources": {
            "num_machines": 3,
            "num_mpiprocs_per_machine": 1,
            "num_cores_per_mpiproc": 1,
        },
    }

    dft_params = {
        "protocol": "debug",
        "cutoff": 300,
    }

    if uks:
        magnetization_per_site = [0 for i in range(len(structure.sites))]
        magnetization_per_site[1] = 1
        magnetization_per_site[2] = -1
        dft_params = {
            "protocol": "debug",
            "uks": uks,
            "magnetization_per_site": magnetization_per_site,
            "charge": 0,
            "periodic": "XYZ",
            "vdw": False,
            "multiplicity": 1,
            "cutoff": 300,
        }

    sys_params = {}

    dft_params["periodic"] = "NONE"
    phonons_params = {"nproc_rep": 1}
    sys_params["colvars"] = "distance atoms 2 3 , distance atoms 1 2"

    builder.structure = structure
    builder.dft_params = Dict(dft_params)
    builder.sys_params = Dict(sys_params)
    builder.phonons_params = Dict(phonons_params)

    _, calc_node = run_get_node(builder)

    assert calc_node.is_finished_ok

    # phonons_out_dict = dict(calc_node.outputs.output_parameters)
    # print()
    # for k in phonons_out_dict:
    #    print(f"  {k}: {phonons_out_dict[k]}")


def example_cp2k_phonons_rks(cp2k_code):
    _example_cp2k_phonons(cp2k_code, "SlabXY", False)


def example_cp2k_phonons_uks(cp2k_code):
    _example_cp2k_phonons(cp2k_code, "SlabXY", True)


if __name__ == "__main__":
    print("#### ", " RKS")
    _example_cp2k_phonons(load_code("cp2k@localhost"), False)

    print("#### ", " UKS")
    _example_cp2k_phonons(load_code("cp2k@localhost"), True)
