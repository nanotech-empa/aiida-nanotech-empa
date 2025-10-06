import typer
from aiida import engine, orm


def cli(cubehandler_code: str):
    builder = orm.load_code(cubehandler_code).get_builder()

    builder.parameters = orm.Dict(
        dict={
            "steps": [
                {
                    "command": "shrink",
                    "args": [
                        "folder1/*ELECTRON*.cube",
                        "folder1/*HART*cube",
                    ],
                    "options": {
                        "output_dir": "out_cubes",
                        "low_precision": True,
                    },
                }
            ]
        }
    )
    builder.parent_folders = {"folder1": orm.load_node(3092)}
    builder.metadata.options = {
        "resources": {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
        },
        "max_wallclock_seconds": 600,
    }

    _, calc_node = engine.run_get_node(builder)

    assert calc_node.is_finished_ok


if __name__ == "__main__":
    typer.run(cli)
