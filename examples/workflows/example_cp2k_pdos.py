import pathlib

import ase.io
import click
from aiida import engine, orm, plugins

Cp2kPdosWorkChain = plugins.WorkflowFactory("nanotech_empa.cp2k.pdos")

DATA_DIR = pathlib.Path(__file__).parent.absolute()
GEO_FILE = "c2h2_on_au111.xyz"


def _example_cp2k_pdos(
    cp2k_code, overlap_code, sc_diag, force_multiplicity, uks, n_nodes, n_cores_per_node
):
    # Check test geometry is already in database.
    qb = orm.QueryBuilder()
    qb.append(orm.Node, filters={"label": {"in": [GEO_FILE]}})
    structure = None
    for node_tuple in qb.iterall():
        node = node_tuple[0]
        structure = node
    if structure is not None:
        print(f"Found existing structure: {structure.pk}")
    else:
        structure = orm.StructureData(ase=ase.io.read(DATA_DIR / GEO_FILE))
        structure.label = GEO_FILE
        structure.store()
        print(f"Created new structure: {structure.pk}")

    builder = Cp2kPdosWorkChain.get_builder()

    builder.metadata.label = "CP2K_PDOS"
    builder.metadata.description = "test description"
    builder.cp2k_code = cp2k_code
    ase_geom_slab = ase.io.read(DATA_DIR / GEO_FILE)
    ase_geom_mol = ase_geom_slab[0:4]
    builder.slabsys_structure = orm.StructureData(ase=ase_geom_slab)
    builder.mol_structure = orm.StructureData(ase=ase_geom_mol)
    builder.pdos_lists = orm.List([("1..4", "molecule"), ("1", "cat")])
    builder.protocol = orm.Str("debug")
    if uks:
        dft_params = {
            "sc_diag": sc_diag,
            "force_multiplicity": force_multiplicity,
            "elpa_switch": False,
            "periodic": "XYZ",
            "uks": uks,
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
            "periodic": "XYZ",
            "uks": uks,
            "smear_t": 150,
        }
    builder.dft_params = orm.Dict(dft_params)

    builder.options = {
        "slab": {
            "max_wallclock_seconds": 600,
            "resources": {
                "num_machines": n_nodes,
                "num_mpiprocs_per_machine": n_cores_per_node,
                "num_cores_per_mpiproc": 1,
            },
        },
        "molecule": {
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
            "--cp2k_input_file1": "parent_slab_folder/aiida.inp",
            "--basis_set_file1": "parent_slab_folder/BASIS_MOLOPT",
            "--xyz_file1": "parent_slab_folder/aiida.coords.xyz",
            "--wfn_file1": "parent_slab_folder/aiida-RESTART.wfn",
            "--emin1": "-2",
            "--emax1": "2",
            "--cp2k_input_file2": "parent_mol_folder/aiida.inp",
            "--basis_set_file2": "parent_mol_folder/BASIS_MOLOPT",
            "--xyz_file2": "parent_mol_folder/aiida.coords.xyz",
            "--wfn_file2": "parent_mol_folder/aiida-RESTART.wfn",
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
        n_nodes=n_nodes,
        n_cores_per_node=n_cores_per_node,
    )


if __name__ == "__main__":
    run_all()
