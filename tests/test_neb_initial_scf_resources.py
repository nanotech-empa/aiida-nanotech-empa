"""Inspect real NEB child builders in isolated storage; never run CP2K."""

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from aiida import engine, orm
from aiida.common.links import LinkType
from aiida.manage import get_manager
from ase import Atoms

from aiida_nanotech_empa.workflows.cp2k import cp2k_utils
from aiida_nanotech_empa.workflows.cp2k.neb_workchain import Cp2kNebWorkChain

# count, BAND machines, ranks/machine, NPROC_REP, SCF machines, SCF ranks/machine.
LAYOUTS = [
    (5, 1, 1, 1, 1, 1),
    (5, 1, 2, 2, 1, 2),
    (5, 1, 4, 4, 1, 4),
    (5, 1, 8, 8, 1, 8),
    (5, 1, 5, 1, 1, 1),
    (5, 1, 10, 2, 1, 2),
    (5, 1, 20, 4, 1, 4),
    (5, 1, 40, 8, 1, 8),
    (6, 1, 8, 4, 1, 4),
    (6, 1, 12, 4, 1, 4),
    (6, 1, 24, 4, 1, 4),
    (6, 1, 4, 4, 1, 4),
    (5, 2, 4, 8, 2, 4),
    (6, 2, 4, 4, 1, 4),
    (5, 5, 4, 4, 1, 4),
    (4, 1, 4, 2, 1, 2),  # Small normal-GUI regression: two replica environments.
    (4, 4, 8, 16, 2, 8),  # Preserve an integral multi-node initial SCF.
    (6, 4, 6, 8, 2, 4),  # A replica environment spans a fractional 6-rank node.
]


@pytest.fixture
def process_factory(aiida_profile_tmp, aiida_localhost, monkeypatch):
    assert aiida_profile_tmp.is_test_profile
    import aiida_nanotech_empa.workflows.cp2k.neb_workchain as tested_module

    assert (
        Path(tested_module.__file__).resolve()
        == Path(__file__).resolve().parents[1]
        / "aiida_nanotech_empa/workflows/cp2k/neb_workchain.py"
    )
    code = orm.InstalledCode(
        computer=aiida_localhost,
        filepath_executable="/bin/false",
        default_calc_job_plugin="cp2k",
    ).store()
    processes = []
    captured = []

    def submit(_, builder):
        captured.append(builder)
        child = orm.CalcJobNode(computer=aiida_localhost).store()
        remote = orm.RemoteData(
            computer=aiida_localhost, remote_path="/tmp/not-executed-initial-scf"
        )
        remote.base.links.add_incoming(
            child, link_type=LinkType.CREATE, link_label="remote_folder"
        )
        remote.store()
        child.set_process_state(engine.ProcessState.FINISHED)
        child.set_exit_status(0)
        return child

    def forbidden_submit(*args, **kwargs):
        raise AssertionError(
            "An allocation regression must never execute a calculation"
        )

    monkeypatch.setattr(Cp2kNebWorkChain, "submit", submit)
    # Resolve the intercepted, already-finished child without an engine loop.
    monkeypatch.setattr(
        Cp2kNebWorkChain,
        "to_context",
        lambda self, **children: self.ctx.update(children),
    )
    monkeypatch.setattr(engine, "submit", forbidden_submit)

    def make(count, resources, nproc, reuse=False, protocol="standard"):
        images = [
            orm.StructureData(
                ase=Atoms(
                    "H2",
                    positions=[[6, 6, 6], [6, 6, 6.74 + i * 0.01]],
                    cell=[12] * 3,
                    pbc=True,
                )
            ).store()
            for i in range(count)
        ]
        scientific = {
            "periodic": "XYZ",
            "poisson_solver": "PERIODIC",
            "charge": 0,
            "uks": False,
            "xc_functional": "PBE",
            "ot_minimizer": "CG",
        }
        options = {
            "resources": resources,
            "max_wallclock_seconds": 600,
            "max_memory_kb": 2048000,
        }
        original = copy.deepcopy(options)
        if reuse:
            # Only availability is substituted: setup runs the real copy-command
            # selection and skips first_scf, as when a compatible WFN is found.
            monkeypatch.setattr(
                cp2k_utils,
                "structure_available_wfn",
                lambda **kwargs: "/tmp/available-existing.wfn",
            )
        process = (
            get_manager()
            .get_runner()
            .instantiate_process(
                Cp2kNebWorkChain,
                code=code,
                structure=images[0],
                replicas={
                    f"replica_{i:03d}": node for i, node in enumerate(images[1:], 1)
                },
                dft_params=orm.Dict(scientific),
                sys_params=orm.Dict({}),
                neb_params=orm.Dict(
                    {
                        "number_of_replica": count,
                        "nproc_rep": nproc,
                        "band_type": "CI-NEB",
                        "k_spring": 0.05,
                        "nsteps_it": 5,
                        "align_frames": "F",
                        "rotate_frames": "F",
                        "optimize_end_points": "F",
                    }
                ),
                protocol=orm.Str(protocol),
                options=options,
            )
        )
        processes.append(process)
        return SimpleNamespace(
            process=process,
            images=images,
            original=original,
            captured=captured,
            scientific=scientific,
        )

    yield make
    for process in processes:
        process.close()


@pytest.mark.parametrize("reuse", [False, True], ids=["missing-wfn", "reused-wfn"])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_setup_and_real_child_builders(process_factory, layout, reuse):
    count, machines, tasks, nproc, scf_machines, scf_tasks = layout
    case = process_factory(
        count,
        {
            "num_machines": machines,
            "num_mpiprocs_per_machine": tasks,
            "num_cores_per_mpiproc": 1,
        },
        nproc,
        reuse,
    )
    process = case.process
    before_attributes = [
        copy.deepcopy(node.base.attributes.all) for node in case.images
    ]
    assert process.setup() is None
    assert process.should_run_scf() is not reuse
    assert process.inputs.options == process.ctx.options == case.original
    expected_band = copy.deepcopy(process.ctx.input_dict)
    assert expected_band["MOTION"]["BAND"]["NPROC_REP"] == nproc
    assert len(expected_band["MOTION"]["BAND"]["REPLICA"]) == count

    if not reuse:
        process.first_scf()
        assert len(case.captured) == 1
        scf = case.captured[0]
        expected_resources = {
            "num_machines": scf_machines,
            "num_mpiprocs_per_machine": scf_tasks,
            "num_cores_per_mpiproc": 1,
        }
        assert scf.metadata.options.resources == expected_resources
        job_resources = scf.code.computer.get_scheduler().create_job_resource(
            **scf.metadata.options.resources
        )
        assert job_resources.get_tot_num_mpiprocs() == nproc
        assert scf.metadata.options.max_wallclock_seconds == 600
        assert scf.metadata.options.max_memory_kb == 2048000
        files, expected, atoms = cp2k_utils.get_dft_inputs(
            case.scientific, case.images[0], "scf_ot_protocol.yml", "standard"
        )
        expected["GLOBAL"]["WALLTIME"] = 600
        assert scf.parameters.get_dict() == expected
        np.testing.assert_array_equal(
            scf.structure.get_ase().positions, atoms.positions
        )
        assert scf.structure.pbc == (True, True, True)
        assert {name: node.get_content() for name, node in scf.file.items()} == {
            name: node.get_content() for name, node in files.items()
        }
    else:
        assert case.captured == []
        assert process.ctx.scf_options == case.original

    process.submit_neb()
    assert len(case.captured) == (1 if reuse else 2)
    band = case.captured[-1]
    assert band.metadata.options.resources == case.original["resources"]
    assert band.parameters.get_dict() == expected_band
    assert band.code.uuid == process.inputs.code.uuid
    assert band.metadata.options.max_wallclock_seconds == 600
    commands = band.metadata.options.prepend_text.splitlines()
    assert len(commands) == count
    assert all(
        ("available-existing.wfn" if reuse else "aiida-RESTART.wfn") in cmd
        for cmd in commands
    )
    assert all(f"aiida-BAND{i}-RESTART.wfn" in cmd for i, cmd in enumerate(commands, 1))
    assert process.inputs.options == process.ctx.options == case.original
    assert [node.base.attributes.all for node in case.images] == before_attributes
    if not reuse:
        case.captured[0].metadata.options.resources["num_machines"] = 99
    process.ctx.scf_options["resources"]["num_machines"] = 88
    assert process.inputs.options == process.ctx.options == case.original
    assert all(
        node.get_job_id() is None
        for node in orm.QueryBuilder().append(orm.CalcJobNode).all(flat=True)
    )


@pytest.mark.parametrize("protocol", ["standard", "low_accuracy"])
def test_scf_retains_threads_and_redundant_resource_fields(process_factory, protocol):
    case = process_factory(
        6,
        {
            "num_machines": 2,
            "num_mpiprocs_per_machine": 6,
            "num_cores_per_mpiproc": 2,
            "num_cores_per_machine": 12,
            "tot_num_mpiprocs": 12,
        },
        4,
        protocol=protocol,
    )
    process = case.process
    assert process.setup() is None
    process.first_scf()
    assert case.captured[0].metadata.options.resources == {
        "num_machines": 1,
        "num_mpiprocs_per_machine": 4,
        "num_cores_per_mpiproc": 2,
        "num_cores_per_machine": 8,
        "tot_num_mpiprocs": 4,
    }
    _, expected, _ = cp2k_utils.get_dft_inputs(
        case.scientific, case.images[0], "scf_ot_protocol.yml", protocol
    )
    expected["GLOBAL"]["WALLTIME"] = 600
    assert case.captured[0].parameters.get_dict() == expected
    assert process.inputs.options == case.original


@pytest.mark.parametrize(
    ("resources", "nproc", "diagnostic"),
    [
        (
            {"num_machines": 1, "num_mpiprocs_per_machine": 4},
            0,
            "nproc_rep must be a positive integer",
        ),
        (
            {"num_machines": 1, "num_mpiprocs_per_machine": 4},
            1.5,
            "nproc_rep must be a positive integer",
        ),
        (
            {"num_machines": 1, "num_mpiprocs_per_machine": 4},
            True,
            "nproc_rep must be a positive integer",
        ),
        (
            {"num_machines": 1, "num_mpiprocs_per_machine": 4},
            3,
            "must divide the total BAND MPI allocation",
        ),
        (
            {"num_machines": 1, "num_mpiprocs_per_machine": 4},
            8,
            "must divide the total BAND MPI allocation",
        ),
        (
            {"num_machines": 0, "num_mpiprocs_per_machine": 4},
            2,
            "num_machines must be a positive integer",
        ),
        (
            {
                "num_machines": 1,
                "num_mpiprocs_per_machine": 4,
                "num_cores_per_mpiproc": 0,
            },
            2,
            "num_cores_per_mpiproc must be a positive integer",
        ),
        (
            {"num_machines": 1, "num_mpiprocs_per_machine": 4, "tot_num_mpiprocs": 6},
            2,
            "tot_num_mpiprocs",
        ),
        (
            {
                "num_machines": 1,
                "num_mpiprocs_per_machine": 4,
                "num_cores_per_machine": 5,
            },
            2,
            "cores per machine must match",
        ),
    ],
)
def test_inconsistent_allocation_returns_explicit_exit(
    process_factory, monkeypatch, resources, nproc, diagnostic
):
    case = process_factory(4, resources, nproc)
    messages = []
    monkeypatch.setattr(case.process, "report", messages.append)
    result = case.process.setup()
    assert result.status == 392
    assert any(diagnostic in message for message in messages)
    assert case.captured == []
    assert case.process.inputs.options == case.original
