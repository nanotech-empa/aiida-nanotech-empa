import pathlib

import click
from aiida import engine, orm, plugins
from ase.io import read

GaussianNicsWorkChain = plugins.WorkflowFactory("nanotech_empa.gaussian.nics")
DATA_DIR = pathlib.Path(__file__).parent.absolute()
GEO_FILE1 = "naphthalene.xyz"
GEO_FILE2 = "C9H7N.xyz"


def _example_gaussian_nics(gaussian_code, opt, geofile, charge=0):
    # Check test geometry is already in database.
    qb = orm.QueryBuilder()
    qb.append(orm.Node, filters={"label": {"==": geofile}})
    structure = None
    for node_tuple in qb.iterall():
        node = node_tuple[0]
        structure = node
    if structure is not None:
        print("found existing structure: ", structure.pk)
    else:
        structure = orm.StructureData(ase=read(DATA_DIR / geofile))
        structure.label = geofile
        structure.store()
        print("created new structure: ", structure.pk)

    builder = GaussianNicsWorkChain.get_builder()
    builder.gaussian_code = gaussian_code
    builder.structure = structure
    builder.charge = orm.Int(charge)
    builder.functional = orm.Str("B3LYP")
    builder.basis_set = orm.Str("6-311G(d,p)")
    builder.opt = orm.Bool(opt)
    builder.options = orm.Dict(
        {
            "resources": {
                "tot_num_mpiprocs": 4,
                "num_machines": 1,
            },
            "max_wallclock_seconds": 1 * 60 * 60,
            "max_memory_kb": 8 * 1024 * 1024,  # GB
        }
    )

    _, wc_node = engine.run_get_node(builder)

    assert wc_node.is_finished_ok
    return wc_node.pk


@click.command("cli")
@click.argument("gaussian_code", default="gaussian@localhost")
def run_nics(gaussian_code):
    # print("#### Running Gaussian NICS WorkChain with geo_opt ####")
    # uuid = _example_gaussian_nics(orm.load_code(gaussian_code),True)
    # print(f"WorkChain completed uuid: {uuid}")
    print(
        f"#### Running Gaussian NICS WorkChain with geo_opt and charge 0 for {GEO_FILE1} ####"
    )
    uuid = _example_gaussian_nics(
        orm.load_code(gaussian_code), True, geofile=GEO_FILE1, charge=0
    )
    print(f"WorkChain completed uuid: {uuid}")
    print(
        f"#### Running Gaussian NICS WorkChain with geo_opt and charge 1 for {GEO_FILE2} ####"
    )
    uuid = _example_gaussian_nics(
        orm.load_code(gaussian_code), True, geofile=GEO_FILE2, charge=1
    )
    print(f"WorkChain completed uuid: {uuid}")


if __name__ == "__main__":
    run_nics()
