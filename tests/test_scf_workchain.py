from types import SimpleNamespace
from uuid import uuid4

import pytest
from aiida import orm, plugins

Cp2kScfWorkChain = plugins.WorkflowFactory("nanotech_empa.cp2k.scf")


@pytest.mark.parametrize(
    ("overlap_matrix", "rejected"),
    [
        ("remote_and_sparse_retrieved", "sparse_overlap_code"),
        ("dense", "overlap_matrix"),
    ],
)
def test_validator_overlap_matrix(overlap_matrix, rejected):
    message = Cp2kScfWorkChain._validate_inputs(
        {"overlap_matrix": orm.Str(overlap_matrix)}, None
    )

    assert rejected in message


@pytest.mark.parametrize(
    ("run_diag_scf", "dft_params", "with_settings", "rejected"),
    [
        # Diagonalization skipped: diag-only inputs are rejected.
        (
            False,
            {"nhomo": 3, "added_mos": 10},
            True,
            ["nhomo", "added_mos", "settings"],
        ),
        # Diagonalization runs: diag-only inputs are accepted.
        (True, {"nhomo": 3, "added_mos": 10}, True, None),
        # Falsy diag-only params have no effect, so they are accepted.
        (False, {"sc_diag": False, "elpa_switch": False}, False, None),
    ],
)
def test_validator_diag_only_inputs(run_diag_scf, dft_params, with_settings, rejected):
    inputs = {
        "run_diag_scf": orm.Bool(run_diag_scf),
        "overlap_matrix": orm.Str("none"),
        "dft_params": orm.Dict(dft_params),
    }
    if with_settings:
        inputs["settings"] = orm.Dict()

    message = Cp2kScfWorkChain._validate_inputs(inputs, None)

    if rejected is None:
        assert message is None
    else:
        for name in rejected:
            assert name in message


@pytest.mark.parametrize(
    ("hook", "run_diag_scf", "overlap_matrix", "printed"),
    [
        # The overlap matrix is printed in the last SCF step only.
        ("update_ot_input_dict", False, "remote_only", True),
        ("update_ot_input_dict", True, "remote_only", False),
        ("update_diag_input_dict", True, "remote_only", True),
        ("update_ot_input_dict", False, "none", False),
        ("update_diag_input_dict", True, "none", False),
    ],
)
def test_overlap_matrix_printed_in_last_scf_step(
    hook, run_diag_scf, overlap_matrix, printed
):
    workchain = SimpleNamespace(
        inputs=SimpleNamespace(
            run_diag_scf=orm.Bool(run_diag_scf),
            overlap_matrix=orm.Str(overlap_matrix),
            overlap_ndigits=orm.Int(10),
        ),
        should_run_bader=lambda: False,
    )
    input_dict = {"FORCE_EVAL": {"DFT": {}}}

    getattr(Cp2kScfWorkChain, hook)(workchain, input_dict)

    if printed:
        ao_matrices = input_dict["FORCE_EVAL"]["DFT"]["PRINT"]["AO_MATRICES"]
        assert ao_matrices["OVERLAP"] == "T"
        assert ao_matrices["NDIGITS"] == 10
    else:
        assert "PRINT" not in input_dict["FORCE_EVAL"]["DFT"]


@pytest.mark.parametrize("run_diag_scf", [False, True])
@pytest.mark.parametrize("already_tagged", [False, True])
def test_finalize_surfaces_outputs(run_diag_scf, already_tagged):
    """Expose OT outputs and append the workflow to the structure's search tag."""
    structure = orm.StructureData(pbc=False)
    structure.append_atom(position=(0.0, 0.0, 0.0), symbols="H")
    structure.store()
    previous_workflows = [str(uuid4())] if already_tagged else []
    if already_tagged:
        structure.base.extras.set("surfaces", previous_workflows)

    ot_scf, diag_scf = (
        SimpleNamespace(
            is_finished_ok=True,
            outputs=SimpleNamespace(
                output_parameters=object(), remote_folder=object(), retrieved=object()
            ),
        )
        for _ in range(2)
    )
    outputs = {}
    workchain = SimpleNamespace(
        node=SimpleNamespace(uuid=str(uuid4())),
        inputs=SimpleNamespace(structure=structure),
        ctx=SimpleNamespace(ot_scf=ot_scf, diag_scf=diag_scf),
        should_run_diag_scf=lambda: run_diag_scf,
        should_run_sparse_overlap=lambda: False,
        should_run_bader=lambda: False,
        out=outputs.__setitem__,
        report=lambda message: None,
    )

    Cp2kScfWorkChain.finalize(workchain)

    final_calc = diag_scf if run_diag_scf else ot_scf
    assert outputs == {
        "output_parameters": final_calc.outputs.output_parameters,
        "remote_folder": final_calc.outputs.remote_folder,
        "retrieved": final_calc.outputs.retrieved,
        "ot_retrieved": ot_scf.outputs.retrieved,
    }
    assert orm.load_node(structure.uuid).base.extras.get("surfaces") == [
        *previous_workflows,
        workchain.node.uuid,
    ]
