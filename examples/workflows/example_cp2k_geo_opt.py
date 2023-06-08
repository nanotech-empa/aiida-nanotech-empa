import pathlib

import ase.io
import click
from aiida import engine, orm, plugins

Cp2kGeoOptWorkChain = plugins.WorkflowFactory("nanotech_empa.cp2k.geo_opt")

DATA_DIR = pathlib.Path(__file__).parent.absolute()
GEOS = ["h2_on_hbn.xyz", "si_bulk.xyz", "c2h2.xyz"]


def _example_cp2k_geo_opt(cp2k_code, sys_type, uks, n_nodes, n_cores_per_node):
    # Check test geometries are already in database.
    qb = orm.QueryBuilder()
    qb.append(
        orm.Node,
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
            structure = orm.StructureData(ase=ase.io.read(DATA_DIR / required))
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
            "num_machines": n_nodes,
            "num_mpiprocs_per_machine": n_cores_per_node,
            "num_cores_per_mpiproc": 1,
        },
    }
    builder.protocol = orm.Str("debug")

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

    if uks:
        magnetization_per_site = [0 for i in range(len(structure.sites))]
        magnetization_per_site[0] = 1
        magnetization_per_site[1] = -1
        dft_params = {
            "uks": uks,
            "magnetization_per_site": magnetization_per_site,
            "charge": 0,
            "periodic": "XYZ",
            "vdw": False,
            "multiplicity": 1,
            "cutoff": 300,
        }
    else:
        dft_params = {"cutoff": 300}

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
    builder.dft_params = orm.Dict(dict=dft_params)
    builder.sys_params = orm.Dict(dict=sys_params)

    _, calc_node = engine.run_get_node(builder)

    assert calc_node.is_finished_ok

    slabopt_out_dict = dict(calc_node.outputs.output_parameters)
    print()
    for k in slabopt_out_dict:
        print(f"  {k}: {slabopt_out_dict[k]}")


def example_cp2k_slab_opt_rks(cp2k_code):
    _example_cp2k_geo_opt(cp2k_code, "SlabXY", False)


def example_cp2k_slab_opt_uks(cp2k_code):
    _example_cp2k_geo_opt(cp2k_code, "SlabXY", True)


@click.command("cli")
@click.argument("cp2k_code", default="cp2k@localhost")
@click.option("-n", "--n-nodes", default=1)
@click.option("-c", "--n-cores-per-node", default=1)
def run_all(cp2k_code, n_nodes, n_cores_per_node):
    for sys_type in ["SlabXY", "Molecule", "Bulk"]:
        print("#### ", sys_type, " RKS")
        _example_cp2k_geo_opt(
            orm.load_code(cp2k_code),
            sys_type=sys_type,
            uks=False,
            n_nodes=n_nodes,
            n_cores_per_node=n_cores_per_node,
        )

        print("#### ", sys_type, " UKS")
        _example_cp2k_geo_opt(
            orm.load_code(cp2k_code),
            sys_type=sys_type,
            uks=True,
            n_nodes=n_nodes,
            n_cores_per_node=n_cores_per_node,
        )


if __name__ == "__main__":
    run_all()
