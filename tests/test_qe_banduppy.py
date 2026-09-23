import inspect

import ase
import banduppy

from aiida import orm, plugins
from aiida.manage import get_manager

from aiida_nanotech_empa.plugins.qe_banduppy import _runner_script

QeBanduppyUnfoldingWorkChain = plugins.WorkflowFactory(
    "nanotech_empa.qe.banduppy_unfolding"
)


def _code(computer, entry_point):
    return orm.InstalledCode(
        label=entry_point.rsplit(".", maxsplit=1)[-1],
        computer=computer,
        filepath_executable="/bin/true",
        default_calc_job_plugin=entry_point,
    ).store()


def test_banduppy_1_api():
    """Test that the installed BandUPpy exposes the APIs used by the runner."""
    generate_parameters = inspect.signature(
        banduppy.Unfolding.generate_SC_Kpts_from_pc_k_path
    ).parameters
    unfold_parameters = inspect.signature(banduppy.Unfolding.Unfold).parameters

    assert "save_kpts" in generate_parameters
    assert "only_unfold_for_kpts_idxs" in unfold_parameters
    assert "only_unfold_band_idx" in unfold_parameters
    assert "qe_keywards" in unfold_parameters
    assert "banduppy.BandStructure" not in _runner_script()
    compile(_runner_script(), "run_banduppy_qe.py", "exec")


def test_prepare_folded_kpoints(aiida_profile, fixture_localhost):
    """Test k-point generation against the released BandUPpy 1 API."""
    structure = orm.StructureData(
        ase=ase.Atoms("C", positions=[[0, 0, 0]], cell=[4, 4, 4], pbc=True)
    )
    process = (
        get_manager()
        .get_runner()
        .instantiate_process(
            QeBanduppyUnfoldingWorkChain,
            pw_code=_code(fixture_localhost, "quantumespresso.pw"),
            banduppy_code=_code(fixture_localhost, "nanotech_empa.qe_banduppy"),
            structure=structure,
            parameters=orm.Dict(dict={"CONTROL": {}, "SYSTEM": {}, "ELECTRONS": {}}),
            template_remote_folder=orm.RemoteData(
                computer=fixture_localhost, remote_path="/tmp"
            ),
            unfolding_parameters=orm.Dict(
                dict={
                    "supercell_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "path": [[0, 0, 0], [0.5, 0, 0]],
                    "labels": ["G", "X"],
                    "npoints_per_segment": 3,
                }
            ),
            pseudos={},
        )
    )

    process.setup()
    process.prepare_folded_kpoints()

    folded = process.ctx.folded_kpoints.get_kpoints()
    primitive = process.ctx.mapping_arrays.get_array("kpoints_pbz_full")
    assert folded.shape[1] == 3
    assert primitive.shape[1] >= 3
    assert len(folded) > 0
    assert len(primitive) > 0
