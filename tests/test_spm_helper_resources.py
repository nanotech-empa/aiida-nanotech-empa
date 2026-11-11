"""Helper allocation follows the parent without running CP2K or SPM tools."""

import copy
import io
from types import SimpleNamespace

import pytest
from aiida import orm
from aiida.common import AttributeDict
from ase import Atoms

from aiida_nanotech_empa.workflows.cp2k import afm_workchain, stm_workchain


@pytest.mark.parametrize("mode", ["AFM", "STM"])
@pytest.mark.parametrize("ranks", [2, 4, 8])
def test_helper_resources_follow_parent(
    aiida_profile_clean, aiida_localhost, mode, ranks
):
    assert aiida_profile_clean.is_test_profile
    aiida_localhost.set_default_mpiprocs_per_machine(36)
    structure = orm.StructureData(
        ase=Atoms("CH", positions=[[6, 6, 6], [7, 6, 6]], cell=[12] * 3, pbc=True)
    ).store()
    remote = orm.RemoteData(
        computer=aiida_localhost, remote_path="/tmp/never-executed-spm"
    ).store()
    plugin = "nanotech_empa.afm" if mode == "AFM" else "nanotech_empa.stm"
    code = orm.InstalledCode(
        computer=aiida_localhost,
        filepath_executable="/bin/false",
        default_calc_job_plugin=plugin,
        label="not-executed-helper",
    ).store()
    options = {
        "resources": {
            "num_machines": 1,
            "num_mpiprocs_per_machine": ranks,
            "num_cores_per_mpiproc": 1,
        },
        "max_wallclock_seconds": 600,
    }
    original = copy.deepcopy(options)
    params = orm.Dict({"--basis_set_file": "BASIS_MOLOPT"})
    inputs = AttributeDict(
        {
            "options": options,
            "structure": structure,
            "spm_code": code,
            "ppafm_code": code,
            "spm_params": params,
            "ppafm_params": params,
        }
    )
    files = {
        name: orm.SinglefileData(io.BytesIO(b"synthetic fixture\n"), filename=name)
        for name in ["geo_no_labels", "pp"]
    }
    context = SimpleNamespace(
        n_atoms=2,
        files=files,
        dft_params={},
        diag_scf=SimpleNamespace(
            is_finished_ok=True, outputs=SimpleNamespace(remote_folder=remote)
        ),
    )
    captured = []
    harness = SimpleNamespace(
        inputs=inputs,
        ctx=context,
        report=lambda text: None,
        to_context=lambda **kwargs: None,
        submit=lambda cls, **kwargs: captured.append((cls, kwargs)),
    )
    cls = (
        afm_workchain.Cp2kAfmWorkChain
        if mode == "AFM"
        else stm_workchain.Cp2kStmWorkChain
    )
    getattr(cls, "run_" + mode.lower())(harness)
    assert len(captured) == 1
    calculation, actual = captured[0]
    assert actual["metadata"]["options"] == original
    assert actual["parent_calc_folder"].uuid == remote.uuid
    assert actual["code"].uuid == code.uuid
    child = calculation.get_builder()
    child._update(actual)
    assert child.metadata.options.resources == original["resources"]
    assert child.metadata.options.max_wallclock_seconds == 600
    assert calculation.spec().inputs["metadata"]["options"]["withmpi"].default == (
        mode == "STM"
    )
    actual["metadata"]["options"]["resources"]["num_machines"] = 99
    assert options == original
    assert aiida_localhost.get_default_mpiprocs_per_machine() == 36
    assert orm.QueryBuilder().append(orm.ProcessNode).count() == 0


@pytest.mark.parametrize("mode", ["AFM", "STM"])
def test_failed_electronic_stage_does_not_submit_helper(mode):
    captured = []
    harness = SimpleNamespace(
        ctx=SimpleNamespace(
            diag_scf=SimpleNamespace(is_finished_ok=False, is_excepted=True)
        ),
        report=lambda text: None,
        exit_codes=SimpleNamespace(ERROR_TERMINATION="failed"),
        submit=lambda *args, **kwargs: captured.append(kwargs),
    )
    cls = (
        afm_workchain.Cp2kAfmWorkChain
        if mode == "AFM"
        else stm_workchain.Cp2kStmWorkChain
    )
    assert getattr(cls, "run_" + mode.lower())(harness) == "failed"
    assert not captured
