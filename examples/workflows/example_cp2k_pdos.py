import os

import ase.io
from aiida.engine import run_get_node
from aiida.orm import Bool, Dict, List, Str, StructureData, load_code
from aiida.plugins import WorkflowFactory

Cp2kPdosWorkChain = WorkflowFactory("nanotech_empa.cp2k.pdos")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
GEO_FILE = "h2_on_hbn.xyz"


def _example_cp2k_pdos(cp2k_code, overlap_code, sc_diag, force_multiplicity):

    builder = Cp2kPdosWorkChain.get_builder()

    builder.metadata.label = "Cp2kPdosWorkChain"
    builder.metadata.description = "test description"
    builder.code = cp2k_code
    ase_geom_slab = ase.io.read(os.path.join(DATA_DIR, GEO_FILE))
    ase_geom_mol = ase_geom_slab[0:1]
    builder.slabsys_structure = StructureData(ase=ase_geom_slab)
    builder.mol_structure = StructureData(ase=ase_geom_mol)
    builder.pdos_list = List([[1], [1, 2]])
    builder.protocol = Str("debug")
    builder.sc_diag = Bool(False)
    builder.force_multiplicity = Bool(True)
    builder.dft_params = Dict(
        dict={"elpa_switch": False, "uks": False, "smear_temperature": 150}
    )
    builder.overlap_code = overlap_code
    builder.overlap_params = Dict(
        dict={
            "--cp2k_input_file1": "parent_slab_folder/aiida.inp",
            "--basis_set_file1": "parent_slab_folder/BASIS_MOLOPT",
            "--xyz_file1": "parent_slab_folder/geom.xyz",
            "--wfn_file1": "parent_slab_folder/aiida-RESTART.wfn",
            "--emin1": "-2",
            "--emax1": "2",
            "--cp2k_input_file2": "parent_mol_folder/aiida.inp",
            "--basis_set_file2": "parent_mol_folder/BASIS_MOLOPT",
            "--xyz_file2": "parent_mol_folder/geom.xyz",
            "--wfn_file2": "parent_mol_folder/aiida-RESTART.wfn",
            "--nhomo2": "2",
            "--nlumo2": "2",
            "--output_file": "./overlap.npz",
            "--eval_region": ["G", "G", "G", "G", "n-3.0_C", "p2.0"],
            "--dx": "0.2",
            "--eval_cutoff": "14.0",
        }
    )
    builder.options = {
        "max_wallclock_seconds": 600,
        "resources": {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 12,
            "num_cores_per_mpiproc": 1,
        },
    }

    builder.protocol = Str("debug")

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
    print("#### no sc_diag")
    _example_cp2k_pdos(
        load_code("cp2k-9.1@daint-mc-em01"), load_code("py_overlap_4576cd@daint-mc-em01"), False, True
    )

    print("#### sc_diag")
    _example_cp2k_pdos(
        load_code("cp2k-9.1@daint-mc-em01"), load_code("py_overlap_4576cd@daint-mc-em01"), True, True
    )
