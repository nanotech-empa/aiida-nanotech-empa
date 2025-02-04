from aiida import orm
from aiida_shell import launch_shell_job


def example_cubehander():
    remote_folder = orm.load_node(1216)
    results, node = launch_shell_job(
        "/users/yaa/miniconda3/envs/cubehandler/bin/cubehandler",
        arguments=["shrink", ".", "out_cubes"],
        metadata={
            "options": {
                "prepend_text": "conda activate cubehandler",
                "use_symlinks": True,
            },
            "computer": orm.load_computer("daint-gpu"),
            "label": "cube-shrink",
        },
        nodes={"remote_previous_job": remote_folder},
        outputs=["out_cubes"],
    )
    print(results)


if __name__ == "__main__":
    example_cubehander()
