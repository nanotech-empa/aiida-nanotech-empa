import pathlib

import ase.io
import click
from aiida import engine, orm, plugins

Cp2kDiagWorkChain = plugins.WorkflowFactory("nanotech_empa.cp2k.diag")

DATA_DIR = pathlib.Path(__file__).parent.absolute()
GEO_FILE = "c2h2.xyz"


def _example_cp2k_diag(
    cp2k_code, sc_diag, force_multiplicity, uks, n_nodes, n_cores_per_node
):
    builder = Cp2kDiagWorkChain.get_builder()

    builder.metadata.label = "Cp2kDiagWorkChain"
    builder.metadata.description = "test description"
    builder.cp2k_code = cp2k_code
    ase_geom = ase.io.read(DATA_DIR / GEO_FILE)
    builder.structure = orm.StructureData(ase=ase_geom)
    builder.settings = orm.Dict(
        {
            "additional_retrieve_list": [
                "aiida.inp",
                "BASIS_MOLOPT",
                "aiida.coords.xyz",
                "aiida-RESTART.wfn",
            ]
        }
    )
    builder.protocol = orm.Str("debug")
    builder.dft_params = orm.Dict(
        {
            "sc_diag": sc_diag,
            "force_multiplicity": force_multiplicity,
            "elpa_switch": False,
            "added_mos": 10,
            "uks": uks,
            "charge": 0,
            "periodic": "NONE",
            "smear_t": 150,
        }
    )
    if uks:
        builder.dft_params = orm.Dict(
            {
                "sc_diag": sc_diag,
                "force_multiplicity": force_multiplicity,
                "elpa_switch": False,
                "added_mos": 10,
                "uks": uks,
                "charge": 0,
                "periodic": "NONE",
                "multiplicity": 1,
                "smear_t": 150,
                "spin_up_guess": [0],
                "spin_dw_guess": [1],
            }
        )
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

    _, calc_node = engine.run_get_node(builder)

    assert calc_node.is_finished_ok


def example_cp2k_diag_no_sc_diag(cp2k_code):
    _example_cp2k_diag(cp2k_code, False, True, False)


def example_cp2k_diag_sc_diag(cp2k_code):
    _example_cp2k_diag(cp2k_code, True, True, True)


@click.command("cli")
@click.argument("cp2k_code", default="cp2k@localhost")
@click.option("-n", "--n-nodes", default=1)
@click.option("-c", "--n-cores-per-node", default=1)
def run_all(cp2k_code, n_nodes, n_cores_per_node):
    print("#### no sc_diag RKS")
    _example_cp2k_diag(
        orm.load_code(cp2k_code),
        sc_diag=False,
        force_multiplicity=True,
        uks=False,
        n_nodes=n_nodes,
        n_cores_per_node=n_cores_per_node,
    )
    print("#### sc_diag RKS")
    _example_cp2k_diag(
        orm.load_code(cp2k_code),
        sc_diag=True,
        force_multiplicity=True,
        uks=False,
        n_nodes=n_nodes,
        n_cores_per_node=n_cores_per_node,
    )
    print("#### no sc_diag UKS no force")
    _example_cp2k_diag(
        orm.load_code(cp2k_code),
        sc_diag=False,
        force_multiplicity=False,
        uks=True,
        n_nodes=n_nodes,
        n_cores_per_node=n_cores_per_node,
    )
    print("#### sc_diag UKS force")
    _example_cp2k_diag(
        orm.load_code(cp2k_code),
        sc_diag=True,
        force_multiplicity=True,
        uks=True,
        n_nodes=n_nodes,
        n_cores_per_node=n_cores_per_node,
    )


if __name__ == "__main__":
    run_all()
