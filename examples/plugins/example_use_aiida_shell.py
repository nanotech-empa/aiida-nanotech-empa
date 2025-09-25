import yaml
from aiida import orm
from aiida_shell import launch_shell_job


def cubehandler_parser(dirpath):
    # Parse the output of the CubeHandler job
    output = orm.Dict(yaml.safe_load((dirpath / "stdout").read_text()))
    return {"results": output}


def example_cubehander():
    remote_folder = orm.load_node(2715)
    code = orm.load_node(2687)
    _, node = launch_shell_job(
        code,
        arguments=["shrink", "-vv", ".", "out_cubes"],
        metadata={
            "options": {
                "prepend_text": "conda activate cubehandler",
                "use_symlinks": True,
            },
            "label": "cube-shrink",
        },
        parser=cubehandler_parser,
        nodes={"remote_previous_job": remote_folder},
        outputs=["out_cubes"],
    )


if __name__ == "__main__":
    example_cubehander()
