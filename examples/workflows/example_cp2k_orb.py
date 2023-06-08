import pathlib

import ase.io
import click
from aiida import engine, orm, plugins

Cp2kOrbiralsWorkChain = plugins.WorkflowFactory("nanotech_empa.cp2k.orbitals")

DATA_DIR = pathlib.Path(__file__).parent.absolute()
GEO_FILE = "c2h2.xyz"


def _example_cp2k_orb(
    cp2k_code, stm_code, sc_diag, force_multiplicity, uks, n_nodes, n_cores_per_node
):
    # Check test geometry is already in database.
    qb = orm.QueryBuilder()
    qb.append(orm.Node, filters={"label": {"in": [GEO_FILE]}})
    structure = None
    for node_tuple in qb.iterall():
        node = node_tuple[0]
        structure = node
    if structure is not None:
        print("found existing structure: ", structure.pk)
    else:
        structure = orm.StructureData(ase=ase.io.read(DATA_DIR / GEO_FILE))
        structure.label = GEO_FILE
        structure.store()
        print("created new structure: ", structure.pk)

    builder = Cp2kOrbiralsWorkChain.get_builder()

    builder.metadata.label = "CP2K_Orbitals"
    builder.metadata.description = "test description"
    builder.cp2k_code = cp2k_code
    builder.structure = structure
    builder.protocol = orm.Str("debug")

    if uks:
        dft_params = {
            "sc_diag": sc_diag,
            "force_multiplicity": force_multiplicity,
            "elpa_switch": False,
            "periodic": "NONE",
            "uks": uks,
            "charge": 0,
            "multiplicity": 1,
            "smear_t": 150,
            "spin_up_guess": [0],
            "spin_dw_guess": [1],
        }
    else:
        dft_params = {
            "sc_diag": sc_diag,
            "force_multiplicity": force_multiplicity,
            "elpa_switch": False,
            "periodic": "NONE",
            "uks": uks,
            "charge": 0,
            "smear_t": 150,
        }

    builder.dft_params = orm.Dict(dft_params)
    builder.options = {
        "max_wallclock_seconds": 600,
        "resources": {
            "num_machines": n_nodes,
            "num_mpiprocs_per_machine": n_cores_per_node,
            "num_cores_per_mpiproc": 1,
        },
    }
    builder.spm_code = stm_code
    parent_dir = "./parent_calc_folder/"
    builder.spm_params = orm.Dict(
        {
            "--cp2k_input_file": parent_dir + "aiida.inp",
            "--basis_set_file": parent_dir + "BASIS_MOLOPT",
            "--xyz_file": parent_dir + "aiida.coords.xyz",
            "--wfn_file": parent_dir + "aiida-RESTART.wfn",
            "--hartree_file": parent_dir + "aiida-HART-v_hartree-1_0.cube",
            "--orb_output_file": "orb.npz",
            "--eval_region": ["G", "G", "G", "G", "n-1.0_C", "p3.5"],
            "--dx": "0.15",
            "--eval_cutoff": "14.0",
            "--extrap_extent": "5",
            "--n_homo": "3",
            "--n_lumo": "3",
            "--orb_heights": ["3"],
            "--orb_isovalues": ["1e-7"],
            "--orb_fwhms": ["0.1"],
            "--p_tip_ratios": 0,
        }
    )

    _, calc_node = engine.run_get_node(builder)

    assert calc_node.is_finished_ok


def example_cp2k_orb_no_sc_diag(cp2k_code, spm_code):
    _example_cp2k_orb(cp2k_code, spm_code, False, True, False)


def example_cp2k_orb_sc_diag(cp2k_code, spm_code):
    _example_cp2k_orb(cp2k_code, spm_code, True, True, True)


@click.command("cli")
@click.argument("cp2k_code", default="cp2k@localhost")
@click.argument("stm_code", default="stm@localhost")
@click.option("-n", "--n-nodes", default=1)
@click.option("-c", "--n-cores-per-node", default=1)
def run_all(cp2k_code, stm_code, n_nodes, n_cores_per_node):
    print("#### sc_diag UKS force")
    _example_cp2k_orb(
        cp2k_code=orm.load_code(cp2k_code),
        stm_code=orm.load_code(stm_code),
        sc_diag=True,
        force_multiplicity=True,
        uks=True,
        n_nodes=n_nodes,
        n_cores_per_node=n_cores_per_node,
    )


if __name__ == "__main__":
    run_all()
