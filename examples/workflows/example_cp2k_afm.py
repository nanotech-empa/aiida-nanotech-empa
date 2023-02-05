import os
import numpy as np

import ase.io
from aiida.engine import run_get_node
from aiida.orm import  Dict,  StructureData, load_code
from aiida.plugins import WorkflowFactory

Cp2kAfmWorkChain = WorkflowFactory("nanotech_empa.cp2k.afm")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
GEO_FILE = "c2h2_on_au111.xyz"


def _example_cp2k_afm(cp2k_code, afm_code1,afm_code2, sc_diag, force_multiplicity,uks):

    builder = Cp2kAfmWorkChain.get_builder()

    builder.metadata.label = "Cp2kAfmWorkChain"
    builder.metadata.description = "test description"
    builder.cp2k_code = cp2k_code
    ase_geom = ase.io.read(os.path.join(DATA_DIR, GEO_FILE))
    builder.structure = StructureData(ase=ase_geom)
    builder.options = Dict(
        dict={
            "resources": {
                "num_machines": 1,
                "num_mpiprocs_per_machine": 1,
            },
            "max_wallclock_seconds": 600,
        }
    )
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

    builder.afm_pp_code = afm_code1
    builder.afm_2pp_code = afm_code2

    cell = ase_geom.cell
    top_z = np.max(ase_geom.positions[:, 2])
    dx = 0.2
    scanminz = 3.5
    scanmaxz = 5.5
    amp = 1.4
    f0 = 22352.5
    paramdata1 = Dict(dict={
        'probeType':    'Otip',
        'charge':       -0.028108681223969645,
        'sigma':        0.7,
        'tip':          's',
        'klat':         0.34901278868090491,
        'krad':         21.913190531846034,
        'r0Probe':      [0.0, 0.0, 2.97],
        'PBC':          'False',
        'gridA':        list(cell[0]),
        'gridB':        list(cell[1]),
        'gridC':        list(cell[2]),
        'scanMin':      [0.0, 0.0, np.round(top_z, 1)+scanminz],
        'scanMax':      [cell[0,0], cell[1,1], np.round(top_z, 1)+scanmaxz],
        'scanStep':     [dx, dx, dx],
        'Amplitude':    amp,
        'f0Cantilever': f0
    })
    paramdata2 = Dict(dict={
        'Catom':        'Ctip',
        'Oatom':        'Otip',
        'ChargeCuUp':   -0.0669933, 
        'ChargeCuDown': -0.0627402, 
        'Ccharge':      0.212718, 
        'Ocharge':      -0.11767,
        'sigma':        0.7,
        'Cklat':        0.24600212465950813,
        'Oklat':        0.15085476515590224,
        'Ckrad':        20,
        'Okrad':        20,
        'rC0':          [0.0, 0.0, 1.82806112489999961213],
        'rO0':          [0.0, 0.0, 1.14881347770000097341],
        'PBC':          'False',
        'gridA':        list(cell[0]),
        'gridB':        list(cell[1]),
        'gridC':        list(cell[2]),
        'scanMin':      [0.0, 0.0, np.round(top_z, 1)+scanminz],
        'scanMax':      [cell[0,0], cell[1,1], np.round(top_z, 1)+scanmaxz],
        'scanStep':     [dx, dx, dx],
        'Amplitude':    amp,
        'f0Cantilever': f0,
        'tip':          'None',
        'Omultipole':   's',
    })
    builder.afm_pp_params = paramdata1
    builder.afm_2pp_params = paramdata2



    _, calc_node = run_get_node(builder)

    assert calc_node.is_finished_ok


def example_cp2k_afm_no_sc_diag(cp2k_code, afm_code1,afm_code2):
    _example_cp2k_afm(cp2k_code,afm_code1,afm_code2, False, True,False)


def example_cp2k_afm_sc_diag(cp2k_code, afm_code):
    _example_cp2k_afm(cp2k_code, afm_code1,afm_code2, True, True,True)


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
    _example_cp2k_afm(
        load_code("cp2k-9.1@daint-mc-em01"), load_code("py_afm_pp_8aa7a6@daint-mc-em01"),load_code("py_afm_2pp_ba8f05@daint-mc-em01"), False, False, True
    )
    #print("#### sc_diag UKS force")
    #_example_cp2k_stm(
    #    load_code("cp2k-9.1@daint-mc-em01"), load_code("py_stm_4576cd@daint-mc-em01"), True, True, True
    #)    