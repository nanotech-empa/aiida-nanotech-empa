"""Benchmark orchestration regressions; no external calculation is launched."""

from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiida import orm
from aiida.common import AttributeDict

from aiida_nanotech_empa.workflows.cp2k.cp2k_benchmark_workchain import (
    Cp2kBenchmarkWorkChain,
    analyze_speedup,
    get_timing_from_FolderData,
)


def test_all_failed_report():
    report = analyze_speedup(orm.Dict(dict={"4_1_1": ["FAILED", "123"]}))
    assert report["min_times_per_nnodes"] == {}
    assert report["closest_to_60"] is None
    assert "No successful" in report["summary"]


def test_timing_metric_and_missing_output():
    folder = orm.FolderData()
    folder.put_object_from_filelike(
        StringIO(
            "  3 OT CG  0.15E+00  4.25  0.0001\n  4 OT LS  0.30E+00  2.75  0.0002\n"
        ),
        "aiida.out",
    )
    # Keep the original OT iteration 3 + 4 elapsed-time metric.
    assert get_timing_from_FolderData(folder).value == 7.0
    assert get_timing_from_FolderData(orm.FolderData()).value == "FAILED"


def test_report_uses_smallest_successful_node_count():
    report = analyze_speedup(
        orm.Dict(
            dict={
                "16_1_1": [40.0, "3"],
                "4_1_1": [100.0, "1"],
                "4_4_1": [120.0, "2"],
                "8_2_1": ["FAILED", "4"],
            }
        )
    ).get_dict()
    assert list(report["min_times_per_nnodes"]) == ["4", "16"]
    assert report["closest_to_60"] == 16


@pytest.mark.parametrize(
    "nodes,ngpus,tasks,threads",
    [([], 1, 2, [1]), ([1], -1, 2, [1]), ([1], 1, 2, [0]), ([2], 4, 2, [1])],
)
def test_invalid_resource_grid(aiida_localhost, nodes, ngpus, tasks, threads):
    aiida_localhost.set_default_mpiprocs_per_machine(8)
    code = orm.InstalledCode(
        computer=aiida_localhost,
        filepath_executable="/bin/true",
        default_calc_job_plugin="cp2k",
    )
    inputs = dict(
        code=code,
        protocol=orm.Str("scf_ot_no_wfn"),
        list_nodes=orm.List(list=nodes),
        ngpus=orm.Int(ngpus),
        max_tasks_per_node=orm.Int(tasks),
        list_threads_per_task=orm.List(list=threads),
        wallclock=orm.Int(600),
        multiplicity=orm.Int(0),
    )
    assert Cp2kBenchmarkWorkChain.validate_inputs(inputs, None)


def test_setup_serializes_numeric_inputs(aiida_localhost):
    aiida_localhost.set_default_mpiprocs_per_machine(8)
    structure = orm.StructureData(cell=[[8, 0, 0], [0, 8, 0], [0, 0, 8]])
    structure.append_atom(position=(0, 0, 0), symbols="H")
    code = orm.InstalledCode(
        computer=aiida_localhost,
        filepath_executable="/bin/true",
        default_calc_job_plugin="cp2k",
    )
    chain = SimpleNamespace(
        inputs=AttributeDict(
            dict(
                code=code,
                structure=structure,
                protocol=orm.Str("scf_ot_no_wfn"),
                cutoff=orm.Int(400),
                multiplicity=orm.Int(2),
                wallclock=orm.Int(1800),
            )
        ),
        ctx=AttributeDict(),
        report=lambda _: None,
    )
    Cp2kBenchmarkWorkChain.setup(chain)
    dft = chain.ctx.input_dict["FORCE_EVAL"]["DFT"]
    assert type(dft["MULTIPLICITY"]) is int
    assert type(dft["MGRID"]["CUTOFF"]) is int


def test_requested_alps_grid(aiida_localhost):
    from itertools import product

    aiida_localhost.set_default_mpiprocs_per_machine(256)
    code = orm.InstalledCode(
        computer=aiida_localhost,
        filepath_executable="/bin/true",
        default_calc_job_plugin="cp2k",
    )
    builder = Cp2kBenchmarkWorkChain.get_builder()
    builder.code = code
    inputs = {
        name: Cp2kBenchmarkWorkChain.spec().inputs[name].default()
        for name in (
            "list_nodes",
            "ngpus",
            "max_tasks_per_node",
            "list_threads_per_task",
        )
    }
    inputs["code"] = code
    grid = Cp2kBenchmarkWorkChain.resource_grid(inputs)
    assert grid == list(product(range(1, 11), [4, 8, 12, 16], [2, 4, 6, 8]))
    assert len(grid) == 160


def test_explicit_tasks_obey_gpu_rule(aiida_localhost):
    aiida_localhost.set_default_mpiprocs_per_machine(256)
    code = orm.InstalledCode(
        computer=aiida_localhost,
        filepath_executable="/bin/true",
        default_calc_job_plugin="cp2k",
    )
    inputs = {
        name: Cp2kBenchmarkWorkChain.spec().inputs[name].default()
        for name in (
            "list_nodes",
            "ngpus",
            "max_tasks_per_node",
            "list_threads_per_task",
            "protocol",
            "multiplicity",
            "wallclock",
        )
    }
    inputs["code"] = code
    inputs["list_nodes"] = orm.List(list=[1])
    inputs["list_tasks_per_node"] = orm.List(list=[4, 6])
    assert "multiples" in Cp2kBenchmarkWorkChain.validate_inputs(inputs, None)
    inputs["list_tasks_per_node"] = orm.List(list=[4, 12])
    assert Cp2kBenchmarkWorkChain.validate_inputs(inputs, None) is None
    assert len(Cp2kBenchmarkWorkChain.resource_grid(inputs)) == 8


def test_child_resources_and_openmp_script(tmp_path, monkeypatch):
    from aiida import engine

    aiida_localhost = orm.Computer(
        label="benchmark-slurm",
        hostname="localhost",
        transport_type="core.local",
        scheduler_type="core.slurm",
        workdir=str(tmp_path),
    ).store()
    aiida_localhost.set_default_mpiprocs_per_machine(256)
    code = orm.InstalledCode(
        computer=aiida_localhost,
        filepath_executable="/bin/true",
        default_calc_job_plugin="cp2k",
        prepend_text="export OMP_NUM_THREADS=7",
    ).store()
    structure = orm.StructureData(cell=[[8, 0, 0], [0, 8, 0], [0, 0, 8]])
    structure.append_atom(position=(0, 0, 0), symbols="H")
    inputs = AttributeDict(
        {
            name: Cp2kBenchmarkWorkChain.spec().inputs[name].default()
            for name in (
                "list_nodes",
                "ngpus",
                "max_tasks_per_node",
                "list_threads_per_task",
                "protocol",
                "multiplicity",
                "wallclock",
            )
        }
    )
    inputs.update(
        code=code,
        structure=structure,
        list_tasks_per_node=orm.List(list=[4, 8, 12, 16]),
    )
    submitted = []

    def capture(builder):
        submitted.append(builder)
        return SimpleNamespace(pk=len(submitted))

    chain = SimpleNamespace(
        inputs=inputs,
        ctx=AttributeDict(),
        report=lambda _: None,
        submit=capture,
        to_context=lambda **kwargs: None,
        resource_grid=Cp2kBenchmarkWorkChain.resource_grid,
    )
    Cp2kBenchmarkWorkChain.setup(chain)
    Cp2kBenchmarkWorkChain.submit_calculations(chain)
    assert len(submitted) == 160
    for builder, (_, _, threads) in zip(
        submitted, Cp2kBenchmarkWorkChain.resource_grid(inputs)
    ):
        assert (
            builder.metadata.options.prepend_text == f"export OMP_NUM_THREADS={threads}"
        )
    builder = submitted[-1]
    assert builder.metadata.options.resources == {
        "num_machines": 10,
        "num_mpiprocs_per_machine": 16,
        "num_cores_per_mpiproc": 8,
    }
    builder.metadata.dry_run = True
    builder.metadata.store_provenance = False
    monkeypatch.chdir(tmp_path)
    _, node = engine.run_get_node(builder)
    script = (
        Path(node.dry_run_info["folder"]) / node.dry_run_info["script_filename"]
    ).read_text()
    assert "--nodes=10" in script
    assert "--ntasks-per-node=16" in script
    assert "--cpus-per-task=8" in script
    assert script.index("export OMP_NUM_THREADS=7") < script.index(
        "export OMP_NUM_THREADS=8"
    )


def test_all_failed_workchain_keeps_outputs(aiida_localhost):
    aiida_localhost.set_default_mpiprocs_per_machine(8)
    code = orm.InstalledCode(
        computer=aiida_localhost,
        filepath_executable="/bin/true",
        default_calc_job_plugin="cp2k",
    )
    structure = orm.StructureData(pbc=False).store()
    inputs = AttributeDict(
        dict(
            code=code,
            structure=structure,
            list_nodes=orm.List(list=[1]),
            list_tasks_per_node=orm.List(list=[4]),
            list_threads_per_task=orm.List(list=[2]),
        )
    )
    outputs = {}
    chain = SimpleNamespace(
        inputs=inputs,
        ctx=AttributeDict(
            dict(
                run_1_4_2=SimpleNamespace(
                    is_finished_ok=False, get_job_id=lambda: "failed-job"
                )
            )
        ),
        node=SimpleNamespace(uuid="00000000-0000-0000-0000-000000000001"),
        report=lambda _: None,
        out=outputs.__setitem__,
        resource_grid=Cp2kBenchmarkWorkChain.resource_grid,
        exit_codes=Cp2kBenchmarkWorkChain.exit_codes,
    )
    result = Cp2kBenchmarkWorkChain.finalize(chain)
    assert result.status == 390
    assert outputs["timings"]["1_4_2"] == ["FAILED", "failed-job"]
    assert outputs["report"]["closest_to_60"] is None


@pytest.mark.parametrize("explicit_tasks", [False, True])
def test_zero_gpus_matches_one_gpu(aiida_localhost, explicit_tasks):
    from itertools import product
    from aiida_nanotech_empa.workflows.cp2k.cp2k_benchmark_workchain import (
        find_multiples_of_ngpus,
    )

    aiida_localhost.set_default_mpiprocs_per_machine(8)
    code = orm.InstalledCode(
        computer=aiida_localhost,
        filepath_executable="/bin/true",
        default_calc_job_plugin="cp2k",
    )
    inputs = {
        name: Cp2kBenchmarkWorkChain.spec().inputs[name].default()
        for name in (
            "protocol",
            "multiplicity",
            "wallclock",
        )
    }
    inputs.update(
        code=code,
        list_nodes=orm.List(list=[1]),
        max_tasks_per_node=orm.Int(4),
        list_threads_per_task=orm.List(list=[1, 2]),
    )
    if explicit_tasks:
        inputs["list_tasks_per_node"] = orm.List(list=[1, 2, 3, 4])
    grids = []
    for gpu_count in (0, 1):
        inputs["ngpus"] = orm.Int(gpu_count)
        assert Cp2kBenchmarkWorkChain.validate_inputs(inputs, None) is None
        grids.append(Cp2kBenchmarkWorkChain.resource_grid(inputs))
    assert grids[0] == grids[1] == list(product([1], [1, 2, 3, 4], [1, 2]))
    assert find_multiples_of_ngpus(0, 1, 4) == [1, 2, 3, 4]
    inputs["ngpus"] = orm.Int(-1)
    assert "non-negative" in Cp2kBenchmarkWorkChain.validate_inputs(inputs, None)


def test_direct_scheduler_rejects_multiple_nodes(aiida_localhost):
    aiida_localhost.set_default_mpiprocs_per_machine(8)
    code = orm.InstalledCode(
        computer=aiida_localhost,
        filepath_executable="/bin/true",
        default_calc_job_plugin="cp2k",
    )
    inputs = {
        name: Cp2kBenchmarkWorkChain.spec().inputs[name].default()
        for name in (
            "protocol",
            "multiplicity",
            "wallclock",
        )
    }
    inputs.update(
        code=code,
        ngpus=orm.Int(0),
        list_nodes=orm.List(list=[1, 2]),
        list_tasks_per_node=orm.List(list=[1]),
        list_threads_per_task=orm.List(list=[2]),
    )
    assert "one machine" in Cp2kBenchmarkWorkChain.validate_inputs(inputs, None)
