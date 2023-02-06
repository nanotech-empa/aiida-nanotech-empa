import os

import ase.io
from aiida.engine import run_get_node
from aiida.orm import  Dict,  StructureData, load_code
from aiida.plugins import WorkflowFactory

Cp2kStmWorkChain = WorkflowFactory("nanotech_empa.cp2k.stm")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
GEO_FILE = "c2h2_on_au111.xyz"


def _example_cp2k_stm(cp2k_code, spm_code, sc_diag, force_multiplicity,uks):

    builder = Cp2kStmWorkChain.get_builder()

    builder.metadata.label = "Cp2kStmWorkChain"
    builder.metadata.description = "test description"
    builder.cp2k_code = cp2k_code
    ase_geom = ase.io.read(os.path.join(DATA_DIR, GEO_FILE))
    builder.structure = StructureData(ase=ase_geom)
    builder.dft_params = Dict(
        dict={"protocol":"debug",
        "sc_diag": sc_diag,
        "force_multiplicity": force_multiplicity,
        "elpa_switch": False, 
        "periodic": 'XYZ',
        "uks": uks, 
        "smear_t": 150}
    )
    if uks:
        builder.dft_params = Dict(
            dict={"protocol":"debug",
            "sc_diag": sc_diag,
            "force_multiplicity": force_multiplicity,
            "elpa_switch": False, 
            "periodic": 'XYZ',
            "uks": uks, 
            "multiplicity":1, 
            "smear_t": 150, 
            "spin_up_guess":[0],
            "spin_dw_guess":[1]}
        )

    builder.options={
            "max_wallclock_seconds": 600,
            "resources": {
                "num_machines": 1,
                "num_mpiprocs_per_machine": 1,
                "num_cores_per_mpiproc": 1,
            }
        }     
    builder.spm_code = spm_code
    parent_dir = "./parent_calc_folder/"
    builder.spm_params = Dict(dict={
            '--cp2k_input_file':    parent_dir+'aiida.inp',
            '--basis_set_file':     parent_dir+'BASIS_MOLOPT',
            '--xyz_file':           parent_dir+'aiida.coords.xyz',
            '--wfn_file':           parent_dir+'aiida-RESTART.wfn',
            '--hartree_file':       parent_dir+'aiida-HART-v_hartree-1_0.cube',
            '--orb_output_file':    'stm.npz',
            '--eval_region':        ['G', 'G', 'G', 'G', 'n-2.0_C', 'p4.0'],
            '--dx':                 '0.15',
            '--eval_cutoff':        '14.0',
            '--extrap_extent':      '5',
            '--energy_range':       ['-1.00', '1.00', '0.5'],
            '--heights':            ['4.0'],
            '--isovalues':          ['1e-7'],
            '--fwhms':              ['0.1'],
            '--p_tip_ratios':       0.0,
        }
    )


    _, calc_node = run_get_node(builder)

    assert calc_node.is_finished_ok


def example_cp2k_stm_no_sc_diag(cp2k_code, spm_code):
    _example_cp2k_stm(cp2k_code,spm_code, False, True,False)


def example_cp2k_stm_sc_diag(cp2k_code, spm_code):
    _example_cp2k_stm(cp2k_code, spm_code, True, True,True)


if __name__ == "__main__":
    #print("#### no sc_diag RKS")
    #_example_cp2k_stm(
    #    load_code("cp2k-9.1@daint-mc-em01"), load_code("py_stm_4576cd@daint-mc-em01"), False, True,False
    #)
    #print("#### sc_diag RKS")
    #_example_cp2k_stm(
    #    load_code("cp2k-9.1@daint-mc-em01"), load_code("py_stm_4576cd@daint-mc-em01"), True, True,False
    #)
    #
    print("#### no sc_diag UKS no force")
    _example_cp2k_stm(
        load_code("cp2k@localhost"), load_code("py_stm_4576cd@localhost"), False, False, True
    )
    #print("#### sc_diag UKS force")
    #_example_cp2k_stm(
    #    load_code("cp2k-9.1@daint-mc-em01"), load_code("py_stm_4576cd@daint-mc-em01"), True, True, True
    #)    