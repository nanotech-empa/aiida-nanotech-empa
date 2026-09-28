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
    (
        "run_diag_scf",
        "overlap_matrix",
        "primitive_vectors",
        "dft_params",
        "window",
        "rejected",
    ),
    [
        # All unfolding requirements are met, with and without an energy window.
        (True, "remote_only", "1 0 0; 0 1 0", {"added_mos": 10}, {}, None),
        (
            True,
            "remote_only",
            "1 0 0; 0 1 0",
            {"added_mos": 10},
            {"unfolding_emin": -2.0, "unfolding_emax": 2.0},
            None,
        ),
        # Unfolding reads the diagonalization WFN and the AO overlap matrix.
        (False, "remote_only", "1 0 0; 0 1 0", {"added_mos": 10}, {}, "run_diag_scf"),
        (True, "none", "1 0 0; 0 1 0", {"added_mos": 10}, {}, "overlap_matrix"),
        (
            True,
            "remote_only",
            None,
            {"added_mos": 10},
            {},
            "unfolding_primitive_vectors",
        ),
        # Unfolding needs a periodic lattice.
        (
            True,
            "remote_only",
            "1 0 0; 0 1 0",
            {"added_mos": 10, "periodic": "NONE"},
            {},
            "periodic",
        ),
        # cp2k-spm-tools needs the LUMO to set its energy reference.
        (True, "remote_only", "1 0 0; 0 1 0", {}, {}, "added_mos"),
        # cp2k-spm-tools ignores a one-sided energy window.
        (
            True,
            "remote_only",
            "1 0 0; 0 1 0",
            {"added_mos": 10},
            {"unfolding_emin": -2.0},
            "unfolding_emax",
        ),
    ],
)
def test_validator_unfolding(
    run_diag_scf, overlap_matrix, primitive_vectors, dft_params, window, rejected
):
    inputs = {
        "run_diag_scf": orm.Bool(run_diag_scf),
        "overlap_matrix": orm.Str(overlap_matrix),
        "unfolding_code": object(),  # the validator only checks presence
        "dft_params": orm.Dict(dft_params),
    }
    if primitive_vectors is not None:
        inputs["unfolding_primitive_vectors"] = orm.Str(primitive_vectors)
    inputs.update({key: orm.Float(bound) for key, bound in window.items()})

    message = Cp2kScfWorkChain._validate_inputs(inputs, None)

    if rejected is None:
        assert message is None
    else:
        assert rejected in message


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
        should_run_unfolding=lambda: False,
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


@pytest.mark.parametrize("path", [None, "", "G-X-M-G", "G-X-A1-Y-G"])
def test_run_unfolding_forwards_optional_path(aiida_localhost, monkeypatch, path):
    from aiida.common import AttributeDict

    from aiida_nanotech_empa.workflows.cp2k import scf_workchain

    code = orm.InstalledCode(
        computer=aiida_localhost,
        filepath_executable="/bin/true",
        default_calc_job_plugin="nanotech_empa.cp2k_unfolding",
    ).store()
    inputs = AttributeDict(
        dict(
            unfolding_code=code,
            unfolding_primitive_vectors=orm.Str("1 0 0; 0 1 0"),
            unfolding_lattice_type=orm.Str("auto"),
            overlap_threshold=orm.Float(1e-10),
        )
    )
    if path is not None:
        inputs.unfolding_path = orm.Str(path)
    submitted = []
    chain = SimpleNamespace(
        inputs=inputs,
        ctx=SimpleNamespace(
            diag_scf=SimpleNamespace(
                outputs=SimpleNamespace(
                    remote_folder=orm.RemoteData(
                        computer=aiida_localhost, remote_path="/tmp/parent"
                    )
                )
            )
        ),
        report=lambda message: None,
        submit=lambda builder: submitted.append(builder),
        _serial_postprocessing_metadata=lambda label: {
            "options": {"resources": {"num_machines": 1}}
        },
    )
    monkeypatch.setattr(
        scf_workchain.common_utils, "check_if_calc_ok", lambda *args: True
    )
    Cp2kScfWorkChain.run_unfolding(chain)
    assert len(submitted) == 1
    if path is None:
        # A default would reach the builder through the spec, not these inputs.
        assert not Cp2kScfWorkChain.spec().inputs["unfolding_path"].has_default()
        assert submitted[0].path is None
    else:
        assert submitted[0].path.value == path
