from aiida import orm, plugins

from aiida_nanotech_empa.workflows.cp2k.cp2k_utils import validate_on_unhandled_failure


def test_validate_on_unhandled_failure_accepts_known_actions():
    for action in ("abort", "pause", "restart_once", "restart_and_pause"):
        assert validate_on_unhandled_failure(orm.Str(action), None) is None


def test_validate_on_unhandled_failure_rejects_unknown_action():
    message = validate_on_unhandled_failure(orm.Str("retry"), None)
    assert message is not None
    assert "retry" in message


def test_validate_on_unhandled_failure_allows_none():
    assert validate_on_unhandled_failure(None, None) is None


def test_geo_opt_restart_policy_inputs_registered():
    Cp2kGeoOptWorkChain = plugins.WorkflowFactory("nanotech_empa.cp2k.geo_opt")
    spec = Cp2kGeoOptWorkChain.spec()
    port = spec.inputs.get_port("on_unhandled_failure")
    assert port.default().value == "pause"
    assert port.validator is not None
