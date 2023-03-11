import os
import ase.io
from ase import Atoms

from aiida.engine import run_get_node
from aiida.orm import Int, List, Str, StructureData, load_code
from aiida.plugins import WorkflowFactory

Cp2kGeoOptWorkChain = WorkflowFactory("nanotech_empa.cp2k.geo_opt")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
GEOS = ["h2_on_hbn.xyz", "si_bulk.xyz", "c2h2.xyz"]


def _example_cp2k_geo_opt(cp2k_code, sys_type, uks):
    # check test geometries are already in database
    qb = QueryBuilder()
    qb.append(
        Node,
        filters={"label": {"in": GEOS}},
    )
    structures = {}
    for node_tuple in qb.iterall():
        node = node_tuple[0]
        structures[node.label] = node
    for required in GEOS:
        if required in structures:
            print("found existing structure: ", required, structures[required].pk)
        else:
            structure = StructureData(ase=ase.io.read(os.path.join(DATA_DIR, required)))
            structure.label = required
            structure.store()
            structures[required] = structure
            print("created new structure: ", required, structure.pk)

    builder = Cp2kGeoOptWorkChain.get_builder()

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

    # define structure
    if sys_type == "SlabXY":
        structure = structures["h2_on_hbn.xyz"]
        builder.metadata.label = "CP2K_GeoOpt"
    elif sys_type == "Molecule":
        structure = structures["c2h2.xyz"]
        builder.metadata.label = "CP2K_GeoOpt"
    elif sys_type == "Bulk":
        structure = structures["si_bulk.xyz"]
        builder.metadata.label = "CP2K_CellOpt"

    dft_params = {
        "protocol": "debug",
        "cutoff": 300,
    }

    if uks:
        magnetization_per_site = [0 for i in range(len(structure.sites))]
        magnetization_per_site[0] = 1
        magnetization_per_site[1] = -1
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

    # adapt parameters to structure
    if sys_type == "SlabXY":
        sys_params[
            "constraints"
        ] = "fixed z 3..18 , collective 1 [ev/angstrom^2] 40 [angstrom] 0.75"
        sys_params["colvars"] = "distance atoms 1 2"
    if sys_type == "Molecule":
        dft_params["periodic"] = "NONE"
        sys_params[
            "constraints"
        ] = "fixed xyz 1 , collective 1 [ev/angstrom^2] 40 [angstrom] 1.36 , collective 2 [ev/angstrom^2] 40 [angstrom] 1.07"
        sys_params["colvars"] = "distance atoms 2 3 , distance atoms 1 2"
    elif sys_type == "Bulk":
        dft_params["protocol"] = "low_accuracy"
        dft_params["periodic"] = "XYZ"
        sys_params["cell_opt"] = ""
        sys_params["symmetry"] = "ORTHORHOMBIC"
        sys_params["cell_opt_constraint"] = "Z"
        sys_params["keep_symmetry"] = ""
        sys_params["constraints"] = "fixed xy 1 , fixed xyz 2"

    builder.structure = structure
    builder.dft_params = Dict(dict=dft_params)
    builder.sys_params = Dict(dict=sys_params)

    _, calc_node = run_get_node(builder)

    assert calc_node.is_finished_ok

    slabopt_out_dict = dict(calc_node.outputs.output_parameters)
    print()
    for k in slabopt_out_dict:
        print(f"  {k}: {slabopt_out_dict[k]}")


def example_cp2k_slab_opt_rks(cp2k_code):
    _example_cp2k_geo_opt(cp2k_code, "SlabXY", False)


def example_cp2k_slab_opt_uks(cp2k_code):
    _example_cp2k_geo_opt(cp2k_code, "SlabXY", True)


if __name__ == "__main__":
    for sys_type in ["SlabXY", "Molecule", "Bulk"]:
        print("#### ", sys_type, " RKS")
        _example_cp2k_geo_opt(load_code("cp2k@localhost"), sys_type, False)

        print("#### ", sys_type, " UKS")
        _example_cp2k_geo_opt(load_code("cp2k@localhost"), sys_type, True)
