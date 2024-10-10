import pathlib

import ase.io
import click
from aiida import engine, orm, plugins

DATA_DIR = pathlib.Path(__file__).parent.absolute()
GEOS = ["c2h2.xyz"]

Cp2kBenchmarkWorkChain = plugins.WorkflowFactory("nanotech_empa.cp2k.benchmark")

def _example_cp2k_benchmark(cp2k_code, nnodes, max_ntasks,nthreads):
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

    builder = Cp2kBenchmarkWorkChain.get_builder()

    builder.metadata.description = "test description"
    builder.code = cp2k_code
    builder.protocol = orm.Str("scf_ot_no_wfn")
    builder.list_nodes = orm.List(list=nnodes)
    builder.max_tasks_per_node = orm.Int(max_ntasks)
    builder.ngpus=orm.Int(1)
    builder.list_threads_per_task = orm.List(list=nthreads)
    builder.metadata.label = "CP2K_Scf"
    builder.structure = structures["c2h2.xyz"]

    _, calc_node = engine.run_get_node(builder)

    assert calc_node.is_finished_ok

@click.command("cli")
@click.argument("cp2k_code", default="cp2k@localhost")
def run_all(cp2k_code):
    print("####    Starting benchmark")
    _example_cp2k_benchmark(
        orm.load_code(cp2k_code),[1],8,[1,2]
    )

if __name__ == "__main__":
    run_all()
