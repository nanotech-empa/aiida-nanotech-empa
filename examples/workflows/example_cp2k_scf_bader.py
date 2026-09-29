"""Run an OT SCF calculation followed by Bader analysis of its charge density."""

import ase
import click
import numpy as np
from aiida import engine, orm, plugins

Cp2kScfWorkChain = plugins.WorkflowFactory("nanotech_empa.cp2k.scf")


def example_cp2k_scf_bader(cp2k_code, bader_code, n_nodes=1, n_cores_per_node=1):
    builder = Cp2kScfWorkChain.get_builder()
    builder.metadata.label = "Cp2kScfWorkChain Bader"
    builder.cp2k_code = cp2k_code
    builder.bader_code = bader_code
    builder.structure = orm.StructureData(
        ase=ase.Atoms(
            "H2", positions=[[4, 4, 3.63], [4, 4, 4.37]], cell=[8, 8, 8], pbc=True
        )
    )
    builder.protocol = orm.Str("debug")
    builder.dft_params = orm.Dict({"periodic": "XYZ", "cutoff": 150})
    builder.options = orm.Dict(
        {
            "max_wallclock_seconds": 600,
            "resources": {
                "num_machines": n_nodes,
                "num_mpiprocs_per_machine": n_cores_per_node,
                "num_cores_per_mpiproc": 1,
            },
        }
    )
    _, node = engine.run_get_node(builder)
    assert node.is_finished_ok
    assert len(node.called) == 2  # OT SCF and Bader, without diagonalization.
    retrieved = node.outputs.bader_retrieved
    assert {"ACF.dat", "AVF.dat", "BCF.dat"} <= set(
        retrieved.base.repository.list_object_names()
    )
    with retrieved.base.repository.open("ACF.dat") as handle:
        charges = np.loadtxt(handle, skiprows=2, max_rows=2)
    assert charges.shape == (2, 7)
    assert np.isfinite(charges).all()
    np.testing.assert_allclose(charges[:, 4].sum(), 2.0, atol=0.02)
    bader = next(
        child for child in node.called if child.process_label == "BaderCalculation"
    )
    assert bader.inputs.parent_calc_folder.uuid == node.outputs.remote_folder.uuid
    assert bader.outputs.retrieved.uuid == retrieved.uuid


@click.command("cli")
@click.argument("cp2k_code", default="cp2k@localhost")
@click.argument("bader_code", default="bader@localhost")
@click.option("-n", "--n-nodes", default=1)
@click.option("-c", "--n-cores-per-node", default=1)
def run_all(cp2k_code, bader_code, n_nodes, n_cores_per_node):
    example_cp2k_scf_bader(
        orm.load_code(cp2k_code),
        orm.load_code(bader_code),
        n_nodes=n_nodes,
        n_cores_per_node=n_cores_per_node,
    )


if __name__ == "__main__":
    run_all()
