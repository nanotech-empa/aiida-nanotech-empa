mport os

import ase.io
import numpy as np
from aiida.engine import run_get_node
from aiida.orm import Bool,Int, Str, StructureData, load_code
from aiida.plugins import WorkflowFactory

import aiida_nanotech_empa.utils.gaussian_wcs_postprocess as pp

GaussianRelaxWorkChain = WorkflowFactory("nanotech_empa.gaussian.relax")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def _example_gaussian_spin(gaussian_code):#, formchk_code, cubegen_code):
    ase_geom = ase.io.read(os.path.join(DATA_DIR, "benzene.xyz"))
    ase_geom.cell = np.diag([10.0, 10.0, 10.0])

    builder = GaussianRelaxWorkChain.get_builder()
    
    
    
    
    _, wc_node = run_get_node(builder)

    assert wc_node.is_finished_ok

    #pp.make_report(wc_node, nb=False, save_image_loc=OUTPUT_DIR)


if __name__ == "__main__":
    _example_gaussian_spin(
        load_code("gaussian@tigu"),
        #load_code("formchk@localhost"),
        #load_code("cubegen@localhost"),
    )
    