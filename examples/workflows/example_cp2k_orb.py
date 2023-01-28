import os

import ase.io
from aiida.engine import run_get_node
from aiida.orm import Bool, Dict, List, Str, StructureData, load_code
from aiida.plugins import WorkflowFactory

Cp2kPdosWorkChain = WorkflowFactory("nanotech_empa.cp2k.pdos")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
GEO_FILE = "h2.xyz"


def _example_cp2k_orb(cp2k_code, stm_code, sc_diag, force_multiplicity,uks):

    builder = Cp2kPdosWorkChain.get_builder()

    builder.metadata.label = "Cp2kPdosWorkChain"
    builder.metadata.description = "test description"
    builder.cp2k_code = cp2k_code
    ase_geom = ase.io.read(os.path.join(DATA_DIR, GEO_FILE))
    builder.structure = StructureData(ase=ase_geom)
    builder.protocol = Str("debug")
    builder.sc_diag = Bool(sc_diag)
    builder.force_multiplicity = Bool(force_multiplicity)
    builder.dft_params = Dict(
        dict={"elpa_switch": False, "uks": uks, "multiplicity":1, "smear_t": 150, "spin_up_guess":[1],"spin_dw_guess":[2]}
    )
    builder.stm_code = stm_code
    builder.stm_params = Dict(dict={
            '--cp2k_input_file':    parent_dir+'aiida.inp',
            '--basis_set_file':     parent_dir+'BASIS_MOLOPT',
            '--xyz_file':           parent_dir+'geom.xyz',
            '--wfn_file':           parent_dir+'aiida-RESTART.wfn',
            '--hartree_file':       parent_dir+'aiida-HART-v_hartree-1_0.cube',
            '--orb_output_file':    'orb.npz',
            '--eval_region':        ['G', 'G', 'G', 'G', 'n-1.0_C', 'p%.1f'%extrap_plane],
            '--dx':                 '0.15',
            '--eval_cutoff':        '16.0',
            '--extrap_extent':      str(extrap_extent),
            '--n_homo':             str(n_homo_inttext.value+2),
            '--n_lumo':             str(n_lumo_inttext.value+2),
            '--orb_heights':        heights_text.value.split(),
            '--orb_isovalues':      isovals_text.value.split(),
            '--orb_fwhms':          fwhms_text.value.split(),
            '--p_tip_ratios':       ptip_floattext.value,
        }
    )

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
    # sc_diag, force_multiplicity,uks
    #print("#### no sc_diag RKS")
    #_example_cp2k_pdos(
    #    load_code("cp2k-9.1@daint-mc-em01"), load_code("py_overlap_4576cd@daint-mc-em01"), False, True,False
    #)
    #print("#### sc_diag RKS")
    #_example_cp2k_pdos(
    #    load_code("cp2k-9.1@daint-mc-em01"), load_code("py_overlap_4576cd@daint-mc-em01"), True, True,False
    #)
    #print("#### no sc_diag UKS no force")
    #_example_cp2k_pdos(
    #    load_code("cp2k-9.1@daint-mc-em01"), load_code("py_overlap_4576cd@daint-mc-em01"), False, False, True
    #)
    print("#### sc_diag UKS force")
    _example_cp2k_pdos(
        load_code("cp2k-9.1@daint-mc-em01"), load_code("py_overlap_4576cd@daint-mc-em01"), True, True, True
    )    