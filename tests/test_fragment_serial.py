"""Real workflow outline/builders with synthetic async children only. No CalcJobs run."""

import asyncio
import copy
from types import SimpleNamespace as NS

import pytest
from aiida import engine, orm, plugins
from ase.build import molecule

from aiida_nanotech_empa.workflows.cp2k.fragment_separation import (
    Cp2kFragmentSeparationWorkChain,
)

pytest_plugins = ["aiida.tools.pytest_fixtures"]
EVENTS = []
PAYLOADS = []
FAIL_STAGE = None
Base = plugins.WorkflowFactory("cp2k.base")


class SyntheticStage(engine.Process):
    _node_class = orm.WorkChainNode

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.expose_inputs(Base)
        spec.input("test_label", valid_type=orm.Str)
        spec.input("test_result", valid_type=orm.Dict)
        spec.input("test_remote", valid_type=orm.RemoteData)
        spec.output("output_parameters", valid_type=orm.Dict)
        spec.output("remote_folder", valid_type=orm.RemoteData)

    async def run(self):
        label = self.inputs.test_label.value
        EVENTS.append(("start", label))
        await asyncio.sleep(0.025)
        self.out("output_parameters", self.inputs.test_result)
        self.out("remote_folder", self.inputs.test_remote)
        EVENTS.append(("end", label))
        return engine.ExitCode(500 if FAIL_STAGE == label else 0)


class SyntheticCube(engine.Process):
    _node_class = orm.WorkChainNode

    @classmethod
    def define(cls, spec):
        super().define(spec)

    async def run(self):
        EVENTS.append(("start", "cube"))
        await asyncio.sleep(0.025)
        EVENTS.append(("end", "cube"))
        return engine.ExitCode(500 if FAIL_STAGE == "cube" else 0)


class SyntheticSubmissions:
    def submit(self, builder):
        # Capture and inspect actual scientific builders, but submit only synthetic WorkChains.
        if "cp2k" not in builder:
            PAYLOADS.append(
                {
                    "name": "cube",
                    "parameters": builder.parameters.get_dict(),
                    "options": dict(builder.metadata.options),
                    "folders": {
                        key: value.get_remote_path()
                        for key, value in builder.parent_folders.items()
                    },
                }
            )
            return super().submit(SyntheticCube)
        cp = builder.cp2k
        atoms = cp.structure.get_ase()
        params = cp.parameters.get_dict()
        fragment = (
            "all"
            if len(atoms) == 6
            else ("water_a" if atoms.positions[0, 0] < 5 else "water_b")
        )
        phase = "scf" if params["GLOBAL"]["RUN_TYPE"] == "ENERGY" else "opt"
        label = phase + "." + fragment
        PAYLOADS.append(
            {
                "name": label,
                "parameters": params,
                "options": dict(cp.metadata.options),
                "positions": atoms.positions.tolist(),
                "cell": atoms.cell.tolist(),
                "pbc": atoms.pbc.tolist(),
                "max_iterations": builder.max_iterations.value,
                "restart_path": cp.parent_calc_folder.get_remote_path()
                if cp.parent_calc_folder is not None
                else None,
            }
        )
        value = {
            "scf.all": -10.0,
            "scf.water_a": -4.0,
            "scf.water_b": -5.0,
            "opt.water_a": -4.1,
            "opt.water_b": -5.2,
        }[label]
        data = builder._inputs(prune=True)
        data["test_label"] = orm.Str(label)
        data["test_result"] = orm.Dict({"energy": value}).store()
        data["test_remote"] = orm.RemoteData(
            computer=self.inputs.code.computer,
            remote_path="/tmp/synthetic-only/" + label,
        ).store()
        return super().submit(SyntheticStage, **data)


class TestWorkflow(SyntheticSubmissions, Cp2kFragmentSeparationWorkChain):
    __test__ = False


@pytest.fixture
def source_inputs(aiida_profile_clean, aiida_localhost):
    assert aiida_profile_clean.is_test_profile
    a = molecule("H2O")
    a.translate([2, 6, 6])
    b = molecule("H2O")
    b.translate([8, 6, 6])
    a += b
    a.cell = [12] * 3
    a.pbc = True
    source = orm.StructureData(ase=a).store()
    code = orm.InstalledCode(
        computer=aiida_localhost,
        filepath_executable="/bin/false",
        default_calc_job_plugin="cp2k",
    ).store()
    cube = orm.InstalledCode(
        computer=aiida_localhost,
        filepath_executable="/bin/false",
        default_calc_job_plugin="nanotech_empa.cubehandler",
    ).store()
    return {
        "code": code,
        "cubehandler_code": cube,
        "structure": source,
        "protocol": orm.Str("standard"),
        "fragments": {
            name: orm.List(values)
            for name, values in {
                "all": list(range(6)),
                "water_a": [0, 1, 2],
                "water_b": [3, 4, 5],
            }.items()
        },
        "fixed_atoms": orm.List([0, 1, 2]),
        "dft_params": orm.Dict(
            {
                "periodic": "XYZ",
                "uks": False,
                "charges": {"all": 0, "water_a": 0, "water_b": 0},
            }
        ),
        "options": {
            name: {
                "resources": {
                    "num_machines": 1,
                    "num_mpiprocs_per_machine": 2,
                    "num_cores_per_mpiproc": 1,
                },
                "max_wallclock_seconds": 600,
            }
            for name in ["all", "water_a", "water_b"]
        },
    }


def run_case(inputs, serial=None, fail=None, cls=TestWorkflow):
    global FAIL_STAGE
    EVENTS.clear()
    PAYLOADS.clear()
    FAIL_STAGE = fail
    kwargs = dict(inputs)
    if serial is not None:
        kwargs["serial_execution"] = orm.Bool(serial)
    outputs, node = engine.run_get_node(cls, **kwargs)
    events = list(EVENTS)
    payloads = copy.deepcopy(PAYLOADS)
    active = peak = 0
    for action, _ in events:
        active += 1 if action == "start" else -1
        peak = max(peak, active)
    assert active == 0
    return outputs, node, events, payloads, peak


@pytest.mark.parametrize("serial", [None, False, True])
def test_real_outline_serializes_only_when_selected(source_inputs, serial):
    inputs = dict(source_inputs)
    inputs["max_iterations"] = orm.Int(1)
    outputs, node, events, payloads, peak = run_case(inputs, serial)
    assert node.is_finished_ok
    assert peak == (1 if serial else 3)
    starts = [label for event, label in events if event == "start"]
    assert starts == [
        "scf.all",
        "scf.water_a",
        "scf.water_b",
        "opt.water_a",
        "opt.water_b",
        "cube",
    ]
    if serial:
        assert events == [
            (event, label) for label in starts for event in ["start", "end"]
        ]
    assert len(node.called_descendants) == 6 and all(
        isinstance(child, orm.WorkChainNode) for child in node.called_descendants
    )
    for row in payloads[:-1]:
        assert row["options"]["resources"] == inputs["options"]["all"]["resources"]
        assert (
            row["max_iterations"] == 1
            and row["parameters"]["GLOBAL"]["WALLTIME"] == 300
        )
    cube = payloads[-1]
    assert cube["options"]["max_wallclock_seconds"] == 600
    assert cube["options"]["resources"] == {
        "num_machines": 1,
        "num_mpiprocs_per_machine": 1,
    }
    assert all(
        path == "/tmp/synthetic-only/scf." + label
        for label, path in cube["folders"].items()
    )
    assert cube["parameters"]["steps"][0]["args"] == [
        "all/aiida-ELECTRON_DENSITY-1_0.cube",
        "1.0",
        "water_a/aiida-ELECTRON_DENSITY-1_0.cube",
        "-1.0",
        "water_b/aiida-ELECTRON_DENSITY-1_0.cube",
        "-1.0",
    ]
    assert outputs["energies"]["unrelaxed_separation_energy"] == pytest.approx(-1.0)
    assert outputs["energies"]["separation_energy"] == pytest.approx(-0.7)


@pytest.mark.parametrize(
    "failed,count",
    [("scf.all", 1), ("scf.water_a", 2), ("opt.water_a", 4), ("cube", 6)],
)
def test_serial_failure_submits_no_later_stage(source_inputs, failed, count):
    _, node, events, _, peak = run_case(source_inputs, True, failed)
    assert (
        node.exit_status == 300
        and len([x for x in events if x[0] == "start"]) == count
        and peak == 1
    )
    assert "energies" not in node.outputs


def test_default_parallel_keeps_default_attempt_limit(source_inputs):
    outputs, node, _, payloads, peak = run_case(source_inputs)
    assert node.is_finished_ok and peak == 3
    assert all(row["max_iterations"] == 5 for row in payloads[:-1])
    assert outputs["energies"]["separation_energy"] == pytest.approx(-0.7)


def test_serial_preserves_implicit_all_and_optional_helper(source_inputs):
    inputs = dict(source_inputs)
    inputs.pop("cubehandler_code")
    inputs["fragments"] = {
        key: value for key, value in inputs["fragments"].items() if key != "all"
    }
    outputs, node, events, _, peak = run_case(inputs, True)
    assert node.is_finished_ok and peak == 1 and len(node.called_descendants) == 5
    assert [label for event, label in events if event == "start"] == [
        "scf.all",
        "scf.water_a",
        "scf.water_b",
        "opt.water_a",
        "opt.water_b",
    ]
    assert outputs["energies"]["separation_energy"] == pytest.approx(-0.7)
    assert not Cp2kFragmentSeparationWorkChain.should_run_serially(NS(inputs={}))
