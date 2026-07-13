import ase

from aiida import orm, plugins
from aiida_nanotech_empa.workflows.cp2k import cp2k_utils

from aiida_nanotech_empa.workflows.cp2k.cp2k_utils import validate_on_unhandled_failure


def test_validate_on_unhandled_failure_accepts_known_actions():
    for action in ("abort", "pause", "restart_once", "restart_and_pause"):
        assert cp2k_utils.validate_on_unhandled_failure(orm.Str(action), None) is None


def test_validate_on_unhandled_failure_rejects_unknown_action():
    message = cp2k_utils.validate_on_unhandled_failure(orm.Str("retry"), None)
    assert message is not None
    assert "retry" in message


def test_validate_on_unhandled_failure_allows_none():
    assert cp2k_utils.validate_on_unhandled_failure(None, None) is None


def test_geo_opt_restart_policy_inputs_registered():
    Cp2kGeoOptWorkChain = plugins.WorkflowFactory("nanotech_empa.cp2k.geo_opt")
    spec = Cp2kGeoOptWorkChain.spec()
    port = spec.inputs.get_port("on_unhandled_failure")
    assert port.default().value == "pause"
    assert port.validator is not None


def test_get_dft_inputs_accepts_plain_dict(aiida_profile):
    """Regression test: get_dft_inputs expects a plain dict, not an orm.Dict node.

    Cp2kReplicaWorkChain used to pass self.inputs.dft_params (an orm.Dict)
    straight through, which raised inside get_dft_inputs (e.g. "vdw" in
    dft_params). Callers must pass dft_params.get_dict() instead.
    """
    structure = orm.StructureData(
        ase=ase.Atoms(
            "H2",
            positions=[[0, 0, 0], [0, 0, 0.74]],
            cell=[10, 10, 10],
            pbc=True,
        )
    )
    dft_params = orm.Dict(dict={"vdw": False, "periodic": "XYZ"}).get_dict()

    files, input_dict, structure_with_tags = cp2k_utils.get_dft_inputs(
        dft_params,
        structure,
        "geo_opt_protocol.yml",
        "standard",
    )

    assert "FORCE_EVAL" in input_dict
