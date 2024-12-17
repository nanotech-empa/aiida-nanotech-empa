import click
from aiida import engine, orm
from aiida.common.datastructures import StashMode


@click.command("cli")
@click.argument("cubehandler_code", default="cubehandler@localhost")
def example_cubehander(cubehandler_code):
    remote_folder = orm.load_node(1038)
    builder = orm.load_code(cubehandler_code).get_builder()

    builder.parameters = orm.Dict(dict={"some": "parameters"})
    builder.parent_calc_folder = remote_folder

    builder.metadata.options = {
        "resources": {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
        },
        "max_wallclock_seconds": 600,
        "stash": {
            "source_list": ["parent_calc_folder/*.cube"],
            "target_base": "/project/s1267/yaa/aiida_stash/",
            "stash_mode": StashMode.COPY.value,
        },
    }

    _, calc_node = engine.run_get_node(builder)

    assert calc_node.is_finished_ok


if __name__ == "__main__":
    example_cubehander()
