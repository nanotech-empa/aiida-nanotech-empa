import pathlib

import ase.io
import click
from aiida import engine, orm, plugins

Cp2kPdosWorkChain = plugins.WorkflowFactory("nanotech_empa.cp2k.pdos")

DATA_DIR = pathlib.Path(__file__).parent.absolute()
GEOS = ["c2h2_on_au111.xyz", "c2h2_for_pdos.xyz"]


def _example_cp2k_pdos(
    cp2k_code, overlap_code, sc_diag, force_multiplicity, uks, n_nodes, n_cores_per_node
):
    # Check test geometry is already in database.
    qb = orm.QueryBuilder()
    qb.append(orm.Node, filters={"label": {"in": GEOS}})
    structures = {}
    for node in qb.all(flat=True):
        structures[node.label] = node
    for required in GEOS:
        if required in structures:
            print("found existing structure: ", required, structures[required].pk)
        else:
            structure = orm.StructureData(ase=ase.io.read(DATA_DIR / required))
            structure.label = required
            structure.store()
            structures[required] = structure
            print("created new structure: ", required, structure.pk)

    builder = Cp2kPdosWorkChain.get_builder()

    builder.metadata.label = "CP2K_PDOS"
    builder.cp2k_code = cp2k_code
    builder.structure = structures["c2h2_on_au111.xyz"]
    builder.fragment_structure = structures["c2h2_for_pdos.xyz"]
    builder.pdos_lists = orm.List([("1..4", "C2H2"), ("1", "C_at")])
    builder.protocol = orm.Str("debug")
    if uks:
        builder.metadata.description = "automatic test PDOS UKS"
        dft_params = {
            "sc_diag": sc_diag,
            "force_multiplicity": force_multiplicity,
            "elpa_switch": False,
            "periodic": "XYZ",
            "uks": uks,
            "multiplicities": {"all": 1, "fragment": 1},
            "magnetization_per_site": {"fragment": [0, 1, -1, 0]},
            "charges": {"all": 0, "fragment": 0},
            "smear_t": 150,
            "spin_up_guess": [0],
            "spin_dw_guess": [1],
        }
    else:
        builder.metadata.description = "automatic test PDOS RKS"
        dft_params = {
            "sc_diag": sc_diag,
            "force_multiplicity": force_multiplicity,
            "elpa_switch": False,
            "periodic": "XYZ",
            "uks": uks,
            "charges": {"all": 0, "fragment": 0},
            "smear_t": 150,
        }
    builder.dft_params = orm.Dict(dft_params)

    builder.options = {
        "all": {
            "max_wallclock_seconds": 600,
            "resources": {
                "num_machines": n_nodes,
                "num_mpiprocs_per_machine": n_cores_per_node,
                "num_cores_per_mpiproc": 1,
            },
        },
        "fragment": {
            "max_wallclock_seconds": 600,
            "resources": {
                "num_machines": n_nodes,
                "num_mpiprocs_per_machine": n_cores_per_node,
                "num_cores_per_mpiproc": 1,
            },
        },
    }

    builder.overlap_code = overlap_code
    builder.overlap_params = orm.Dict(
        {
            "--cp2k_input_file1": "parent_all_folder/aiida.inp",
            "--basis_set_file1": "parent_all_folder/BASIS_MOLOPT",
            "--xyz_file1": "parent_all_folder/aiida.coords.xyz",
            "--wfn_file1": "parent_all_folder/aiida-RESTART.wfn",
            "--emin1": "-2",
            "--emax1": "2",
            "--cp2k_input_file2": "parent_fragment_folder/aiida.inp",
            "--basis_set_file2": "parent_fragment_folder/BASIS_MOLOPT",
            "--xyz_file2": "parent_fragment_folder/aiida.coords.xyz",
            "--wfn_file2": "parent_fragment_folder/aiida-RESTART.wfn",
            "--nhomo2": "2",
            "--nlumo2": "2",
            "--output_file": "./overlap.npz",
            "--eval_region": ["G", "G", "G", "G", "n-3.0_C", "p2.0"],
            "--dx": "0.2",
            "--eval_cutoff": "14.0",
        }
    )

    _, calc_node = engine.run_get_node(builder)

    assert calc_node.is_finished_ok


def example_cp2k_pdos_no_sc_diag(cp2k_code, overlap_code):
    _example_cp2k_pdos(cp2k_code, overlap_code, False, True)


def example_cp2k_pdos_sc_diag(cp2k_code, overlap_code):
    _example_cp2k_pdos(cp2k_code, overlap_code, True, True)


@click.command("cli")
@click.argument("cp2k_code", default="cp2k@localhost")
@click.argument("overlap_code", default="overlap@localhost")
@click.option("-n", "--n-nodes", default=1)
@click.option("-c", "--n-cores-per-node", default=1)
def run_all(cp2k_code, overlap_code, n_nodes, n_cores_per_node):
    print("#### no sc_diag RKS")
    _example_cp2k_pdos(
        orm.load_code(cp2k_code),
        orm.load_code(overlap_code),
        sc_diag=False,
        force_multiplicity=True,
        uks=False,
        n_nodes=1,
        n_cores_per_node=1,
    )
    print("#### sc_diag UKS magnetization guess only on fragment")
    _example_cp2k_pdos(
        orm.load_code(cp2k_code),
        orm.load_code(overlap_code),
        sc_diag=True,
        force_multiplicity=True,
        uks=True,
        n_nodes=1,
        n_cores_per_node=1,
    )


if __name__ == "__main__":
    run_all()
