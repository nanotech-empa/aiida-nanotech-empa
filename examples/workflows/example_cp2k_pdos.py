import os

import ase.io
from aiida.engine import run_get_node
from aiida.orm import Bool, Dict, List, Str, StructureData, load_code
from aiida.plugins import WorkflowFactory

Cp2kPdosWorkChain = WorkflowFactory("nanotech_empa.cp2k.pdos")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
GEO_FILE = "c2h2_on_au111.xyz"


def _example_cp2k_pdos(cp2k_code, overlap_code, sc_diag, force_multiplicity, uks):
    # check test geometry is already in database
    qb = QueryBuilder()
    qb.append(Node, filters={"label": {"in": [GEO_FILE]}})
    structure = None
    for node_tuple in qb.iterall():
        node = node_tuple[0]
        structure = node
    if structure is not None:
        print("found existing structure: ", structure.pk)
    else:
        structure = StructureData(ase=ase.io.read(os.path.join(DATA_DIR, GEO_FILE)))
        structure.label = GEO_FILE
        structure.store()
        print("created new structure: ", structure.pk)

    builder = Cp2kPdosWorkChain.get_builder()

    builder.metadata.label = "CP2K_PDOS"
    builder.metadata.description = "test description"
    builder.cp2k_code = cp2k_code
    ase_geom_slab = ase.io.read(os.path.join(DATA_DIR, GEO_FILE))
    ase_geom_mol = ase_geom_slab[0:4]
    builder.slabsys_structure = StructureData(ase=ase_geom_slab)
    builder.mol_structure = StructureData(ase=ase_geom_mol)
    builder.pdos_lists = List([("1..4", "molecule"), ("1", "cat")])
    builder.dft_params = Dict(
        {
            "protocol": "debug",
            "sc_diag": sc_diag,
            "force_multiplicity": force_multiplicity,
            "elpa_switch": False,
            "periodic": "XYZ",
            "uks": uks,
            "smear_t": 150,
        }
    )
    if uks:
        builder.dft_params = Dict(
            {
                "protocol": "debug",
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
        )

    builder.options = {
        "slab": {
            "max_wallclock_seconds": 600,
            "resources": {
                "num_machines": 1,
                "num_mpiprocs_per_machine": 1,
                "num_cores_per_mpiproc": 1,
            },
        },
        "molecule": {
            "max_wallclock_seconds": 600,
            "resources": {
                "num_machines": 1,
                "num_mpiprocs_per_machine": 1,
                "num_cores_per_mpiproc": 1,
            },
        },
    }

    builder.overlap_code = overlap_code
    builder.overlap_params = Dict(
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

    _, calc_node = run_get_node(builder)

    assert calc_node.is_finished_ok

    # slabopt_out_dict = dict(calc_node.outputs.output_parameters)
    # print()
    # for k in slabopt_out_dict:
    #    print(f"  {k}: {slabopt_out_dict[k]}")


def example_cp2k_pdos_no_sc_diag(cp2k_code, overlap_code):
    _example_cp2k_pdos(cp2k_code, overlap_code, False, True)


def example_cp2k_pdos_sc_diag(cp2k_code, overlap_code):
    _example_cp2k_pdos(cp2k_code, overlap_code, True, True)


if __name__ == "__main__":
    print("#### no sc_diag RKS")
    _example_cp2k_pdos(
        load_code("cp2k@localhost"),
        load_code("py_overlap_4576cd@localhost"),
        False,
        True,
        False,
    )
    # print("#### sc_diag RKS")
    # _example_cp2k_pdos(
    #    load_code("cp2k-9.1@daint-mc-em01"), load_code("py_overlap_4576cd@daint-mc-em01"), True, True,False
    # )
    # print("#### no sc_diag UKS no force")
    # _example_cp2k_pdos(
    #    load_code("cp2k-9.1@daint-mc-em01"), load_code("py_overlap_4576cd@daint-mc-em01"), False, False, True
    # )
    # print("#### sc_diag UKS force")
    # _example_cp2k_pdos(
    #    load_code("cp2k@localhost"), load_code("py_overlap_4576cd@localhost"), True, True, True
    # )
