import os

import ase.io
import numpy as np
from aiida.engine import run_get_node
from aiida.orm import List, Str, StructureData, load_code
from aiida.plugins import WorkflowFactory

import aiida_nanotech_empa.utils.gaussian_wcs_postprocess as pp

GaussianSpinWorkChain = WorkflowFactory("nanotech_empa.gaussian.spin")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def _example_gaussian_spin(gaussian_code, formchk_code, cubegen_code):
    ase_geom = ase.io.read(os.path.join(DATA_DIR, "benzene-diradical.xyz"))
    ase_geom.cell = np.diag([10.0, 10.0, 10.0])

    builder = GaussianSpinWorkChain.get_builder()
    builder.gaussian_code = gaussian_code
    builder.formchk_code = formchk_code
    builder.cubegen_code = cubegen_code
    builder.structure = StructureData(ase=ase_geom)
    builder.functional = Str("B3LYP")
    builder.empirical_dispersion = Str("GD3")
    builder.basis_set_opt = Str("STO-3G")
    builder.basis_set_scf = Str("STO-3G")
    builder.multiplicity_list = List([0, 1, 3])

    _, wc_node = run_get_node(builder)

    assert wc_node.is_finished_ok

    pp.make_report(wc_node, nb=False, save_image_loc=OUTPUT_DIR)


if __name__ == "__main__":
    _example_gaussian_spin(
        load_code("gaussian@localhost"),
        load_code("formchk@localhost"),
        load_code("cubegen@localhost"),
    )
