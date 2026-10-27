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


def test_primitive_reference_keeps_periodicity_and_collapses_replicas(
    aiida_profile, fixture_localhost
):
    """A two-cell supercell produces one primitive site and the right mesh."""
    import numpy as np

    structure = orm.StructureData(
        ase=ase.Atoms(
            "C2",
            positions=[[0, 0, 0], [4, 0, 0]],
            cell=[8, 4, 4],
            pbc=[True, True, False],
        )
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
                    "supercell_matrix": [[2, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "path": [[0, 0, 0], [0.5, 0, 0], [0, 0, 0]],
                    "labels": ["G", "X", "G"],
                    "npoints_per_segment": [3, 3],
                }
            ),
            pseudos={},
        )
    )
    try:
        process.setup()
        process.prepare_folded_kpoints()
        process.prepare_reference_inputs()
        reference = process.ctx.reference_structure
        np.testing.assert_allclose(reference.cell, np.diag([4, 4, 4]))
        assert reference.pbc == (True, True, False)
        assert len(reference.sites) == 1
        assert reference.kinds[0].symbols == ("C",)
        mesh, offset = process.ctx.reference_scf_kpoints.get_kpoints_mesh()
        assert tuple(mesh) == (2, 1, 1)
        assert tuple(offset) == (0.0, 0.0, 0.0)
        np.testing.assert_allclose(
            process.ctx.reference_kpoints.get_kpoints()[[0, -1]], [[0, 0, 0], [0, 0, 0]]
        )
        assert [label for _, label in process.ctx.reference_kpoints.labels] == [
            "G",
            "X",
            "G",
        ]
    finally:
        process.close()


def test_reference_parameters_are_pristine_without_mutating_template():
    """The primitive reference removes defect charge/spin and rescales bands."""
    import copy

    import numpy as np

    from aiida_nanotech_empa.workflows.qe.banduppy import _reference_qe_parameters

    parameters = {
        "CONTROL": {
            "calculation": "scf",
            "restart_mode": "restart",
            "prefix": "source",
        },
        "SYSTEM": {
            "nbnd": 13,
            "tot_charge": 1,
            "nspin": 2,
            "starting_magnetization": {"C": 0.5},
            "starting_ns_eigenvalue(1,1,1)": 1,
        },
        "ELECTRONS": {
            "startingpot": "file",
            "startingwfc": "file",
            "diago_full_acc": True,
            "conv_thr": 1e-8,
        },
    }
    original = copy.deepcopy(parameters)
    matrix = np.diag([2, 1, 1])
    scf = _reference_qe_parameters(parameters, matrix, calculation="scf")
    bands = _reference_qe_parameters(parameters, matrix, calculation="bands")
    assert parameters == original
    assert scf["CONTROL"] == {
        "calculation": "scf",
        "restart_mode": "from_scratch",
        "prefix": "source",
    }
    assert scf["SYSTEM"] == {}
    assert scf["ELECTRONS"] == {"conv_thr": 1e-8}
    assert bands["SYSTEM"] == {"nbnd": 7}
    assert bands["CONTROL"]["calculation"] == "bands"
    assert bands["CONTROL"]["restart_mode"] == "from_scratch"


def test_folded_diagonalization_preserves_template():
    from aiida_nanotech_empa.workflows.qe.banduppy import _folded_qe_parameters

    parameters = {
        "CONTROL": {"calculation": "scf"},
        "ELECTRONS": {"diagonalization": "david"},
    }
    inherited = _folded_qe_parameters(parameters, {"folded_diagonalization": "inherit"})
    overridden = _folded_qe_parameters(parameters, {"folded_diagonalization": "cg"})
    assert parameters["CONTROL"]["calculation"] == "scf"
    assert parameters["ELECTRONS"]["diagonalization"] == "david"
    assert inherited["CONTROL"]["calculation"] == "bands"
    assert inherited["ELECTRONS"]["diagonalization"] == "david"
    assert overridden["ELECTRONS"]["diagonalization"] == "cg"
