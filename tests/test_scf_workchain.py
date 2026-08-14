import numpy as np
from ase import Atoms
from aiida import engine, orm, plugins

Cp2kScfWorkChain = plugins.WorkflowFactory("nanotech_empa.cp2k.scf")


def _methane_structure():
    center = np.array([4.0, 4.0, 4.0])
    delta = 1.09 / np.sqrt(3.0)
    offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [delta, delta, delta],
            [delta, -delta, -delta],
            [-delta, delta, -delta],
            [-delta, -delta, delta],
        ]
    )
    atoms = Atoms(
        "CH4",
        positions=center + offsets,
        cell=[8.0, 8.0, 8.0],
        pbc=True,
    )
    return orm.StructureData(ase=atoms)


def test_retrieve_sparse_overlap_requires_sparse_overlap_code():
    message = Cp2kScfWorkChain._validate_inputs(
        {"retrieve_sparse_overlap": orm.Bool(True)}, None
    )

    assert "sparse_overlap_code" in message


def test_optional_postprocessing_inputs_are_validated():
    base_inputs = {"retrieve_sparse_overlap": orm.Bool(False)}

    message = Cp2kScfWorkChain._validate_inputs(
        {**base_inputs, "compute_bader_charges": orm.Bool(True)}, None
    )
    assert "bader_code" in message

    unfolding_inputs = {
        **base_inputs,
        "compute_unfolding": orm.Bool(True),
    }
    message = Cp2kScfWorkChain._validate_inputs(unfolding_inputs, None)
    assert "unfolding_code" in message

    unfolding_inputs["unfolding_code"] = object()
    message = Cp2kScfWorkChain._validate_inputs(unfolding_inputs, None)
    assert "unfolding_primitive_vectors" in message

    unfolding_inputs["unfolding_primitive_vectors"] = orm.Str("1 0 0; 0 1 0")
    message = Cp2kScfWorkChain._validate_inputs(unfolding_inputs, None)
    assert "unfolding_primitive_basis_atoms" in message


def test_cp2k_scf_workchain_retrieves_sparse_overlap(
    aiida_profile, cp2k_code, sparse_overlap_code
):
    builder = Cp2kScfWorkChain.get_builder()
    builder.metadata.label = "Cp2kScfWorkChain CH4 sparse overlap test"
    builder.cp2k_code = cp2k_code
    builder.sparse_overlap_code = sparse_overlap_code
    builder.structure = _methane_structure()
    builder.protocol = orm.Str("debug")
    builder.dft_params = orm.Dict(dict={"periodic": "XYZ", "cutoff": 150})
    builder.options = orm.Dict(
        dict={
            "max_wallclock_seconds": 600,
            "resources": {
                "num_machines": 1,
                "num_mpiprocs_per_machine": 1,
                "num_cores_per_mpiproc": 1,
            },
        }
    )
    builder.retrieve_sparse_overlap = orm.Bool(True)
    builder.overlap_threshold = orm.Float(1.0e-10)

    _, node = engine.run_get_node(builder)

    assert node.is_finished_ok
    retrieved = node.outputs.sparse_overlap_retrieved
    assert "sparse_overlap.npz" in retrieved.base.repository.list_object_names()

    with retrieved.base.repository.open("sparse_overlap.npz", "rb") as handle:
        with np.load(handle) as sparse_overlap:
            shape = tuple(sparse_overlap["shape"])
            elements = set(sparse_overlap["element"].astype(str))
            assert shape[0] == shape[1]
            assert shape[0] == sparse_overlap["basis_index"].size
            assert sparse_overlap["data"].size >= shape[0]
            assert {"C", "H"} <= elements
