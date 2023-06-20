"""pytest fixtures for simplified testing."""
import shutil
import subprocess

import pytest
from aiida.common import exceptions
from aiida.orm import Code, Computer, QueryBuilder

pytest_plugins = ["aiida.manage.tests.pytest_fixtures"]


@pytest.fixture(scope="session", autouse=True)
def setup_sssp_pseudos(aiida_profile):
    """Create an SSSP pseudo potential family from scratch."""
    subprocess.run(
        [
            "aiida-pseudo",
            "install",
            "sssp",
            "-p",
            "efficiency",
            "-x",
            "PBE",
            "-v",
            "1.2",
        ]
    )


@pytest.fixture
def fixture_localhost(aiida_localhost):
    """Return a localhost `Computer`."""
    localhost = aiida_localhost
    localhost.set_default_mpiprocs_per_machine(1)
    # By default, the codes are called with "mpirun -np N ..."
    # Disable this, as multiple processes are not supported by e.g. cp2k.ssmp
    localhost.set_mpirun_command([])
    return localhost


@pytest.fixture(scope="function")
def local_code_factory(fixture_localhost):
    """Modified version of aiida_local_code_factory, that uses fixture_localhost"""

    def get_code(
        entry_point, executable, label=None, prepend_text=None, append_text=None
    ):
        if label is None:
            label = executable

        computer = fixture_localhost

        builder = QueryBuilder().append(
            Computer, filters={"uuid": computer.uuid}, tag="computer"
        )
        builder.append(
            Code,
            filters={"label": label, "attributes.input_plugin": entry_point},
            with_computer="computer",
        )

        try:
            code = builder.one()[0]
        except (exceptions.MultipleObjectsError, exceptions.NotExistent):
            code = None
        else:
            return code

        executable_path = shutil.which(executable)
        if not executable_path:
            raise ValueError(
                f'The executable "{executable}" was not found in the $PATH.'
            )

        code = Code(
            input_plugin_name=entry_point,
            remote_computer_exec=[computer, executable_path],
        )
        code.label = label
        code.description = label

        if prepend_text is not None:
            code.set_prepend_text(prepend_text)

        if append_text is not None:
            code.set_append_text(append_text)

        return code.store()

    return get_code


@pytest.fixture(scope="function")
def qe_pw_code(local_code_factory):
    return local_code_factory("quantumespresso.pw", "pw.x")


@pytest.fixture(scope="function")
def qe_pp_code(local_code_factory):
    return local_code_factory("quantumespresso.pp", "pp.x")


@pytest.fixture(scope="function")
def qe_projwfc_code(local_code_factory):
    return local_code_factory("quantumespresso.projwfc", "projwfc.x")


@pytest.fixture(scope="function")
def cp2k_code(local_code_factory):
    prepend_text = "export OMP_NUM_THREADS=2"
    return local_code_factory("cp2k", "cp2k.ssmp", prepend_text=prepend_text)
