import pathlib

import ase.io
import click
from aiida import engine, orm, plugins

StructureData = plugins.DataFactory("core.structure")
Cp2kFragmentSeparationWorkChain = plugins.WorkflowFactory(
    "nanotech_empa.cp2k.fragment_separation"
)


DATA_DIR = pathlib.Path(__file__).parent.absolute()
GEO_FILE = "h2_on_hbn.xyz"


def _example_cp2k_ads_ene(cp2k_code, mult, n_nodes, n_cores_per_node):
    """Example of running a workflow to compute the adsorption energy of a molecule on substrate."""
    # Check test geometry is already in database.
    qb = orm.QueryBuilder()
    qb.append(orm.Node, filters={"label": {"in": [GEO_FILE]}})
    structure = None
    for node_tuple in qb.iterall():
        node = node_tuple[0]
        structure = node
    if structure is not None:
        print(f"Found existing structure: {structure.pk}")
    else:
        structure = StructureData(ase=ase.io.read(DATA_DIR / GEO_FILE))
        structure.label = GEO_FILE
        structure.store()
        print(f"Created new structure: {structure.pk}")
    builder = Cp2kFragmentSeparationWorkChain.get_builder()

    builder.metadata.label = "CP2K_AdsorptionE"
    builder.metadata.description = "h2 on hbn slab"
    builder.code = cp2k_code
    builder.structure = structure
    builder.fixed_atoms = orm.List(
        list=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    )

    builder.fragments = {
        "molecule": orm.List(list=[0, 1]),
        "slab": orm.List(list=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]),
    }

    mag = []
    uks = False
    if mult == 1:
        uks = True
        mag = [0 for i in structure.get_ase()]
        mag[0] = 1
        mag[1] = -1

    dft_params = {
        "uks": uks,
        "multiplicities": {
            "all": mult,
            "molecule": mult,
            "slab": mult,
        },
        "magnetization_per_site": mag,
    }

    builder.dft_params = orm.Dict(dft_params)
    builder.protocol = orm.Str("debug")

    builder.options = {
        "all": {
            "max_wallclock_seconds": 1200,
            "resources": {
                "num_machines": n_nodes,
                "num_mpiprocs_per_machine": n_cores_per_node,
            },
        },
        "molecule": {
            "max_wallclock_seconds": 1200,
            "resources": {
                "num_machines": n_nodes,
                "num_mpiprocs_per_machine": n_cores_per_node,
            },
        },
        "slab": {
            "max_wallclock_seconds": 1200,
            "resources": {
                "num_machines": n_nodes,
                "num_mpiprocs_per_machine": n_cores_per_node,
            },
        },
    }

    _, calc_node = engine.run_get_node(builder)

    assert calc_node.is_finished_ok

    adsorption_energy_out_dict = dict(calc_node.outputs.energies)

    for k in adsorption_energy_out_dict:
        print(f"{k}: {adsorption_energy_out_dict[k]}")


def example_cp2k_ads_ene_rks(cp2k_code):
    _example_cp2k_ads_ene(cp2k_code, 0)


def example_cp2k_slabopt_uks(cp2k_code):
    _example_cp2k_ads_ene(cp2k_code, 1)


@click.command("cli")
@click.argument("cp2k_code", default="cp2k@localhost")
@click.option("-n", "--n-nodes", default=1)
@click.option("-c", "--n-cores-per-node", default=1)
def run_all(cp2k_code, n_nodes, n_cores_per_node):
    print("#### RKS")
    _example_cp2k_ads_ene(
        orm.load_code(cp2k_code),
        mult=0,
        n_nodes=n_nodes,
        n_cores_per_node=n_cores_per_node,
    )

    print("#### UKS")
    _example_cp2k_ads_ene(
        orm.load_code(cp2k_code),
        mult=1,
        n_nodes=n_nodes,
        n_cores_per_node=n_cores_per_node,
    )


if __name__ == "__main__":
    run_all()
