import os
import numpy as np

import ase.io

from aiida.orm import StructureData, Str, Dict, Int
from aiida.orm import load_code
from aiida.engine import submit
from aiida.plugins import WorkflowFactory

GaussianVibronicStmWorkChain = WorkflowFactory(
    'nanotech_empa.gaussian.vibronic_stm')

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def _example_gaussian_spin(gaussian_code):

    ase_geom = ase.io.read(os.path.join(DATA_DIR, "big_molecule.xyz"))
    ase_geom.cell = np.diag([20.0, 20.0, 20.0])

    builder = GaussianVibronicStmWorkChain.get_builder()
    builder.gaussian_code = gaussian_code
    builder.structure = StructureData(ase=ase_geom)
    builder.functional = Str('b3pw91')
    builder.empirical_dispersion = Str('GD3')
    builder.basis_set = Str('6-31g(d,p)')
    builder.charge_final = Int(1)

    builder.options = Dict(
        dict={
            "resources": {
                "num_machines": 1,
                "tot_num_mpiprocs": 8,
            },
            "max_memory_kb": int(1.25 * 40) * 1024 * 1024,
            "max_wallclock_seconds": 60 * 60 * 24,
        })

    #_, wc_node = run_get_node(builder)
    submit(builder)

    #assert wc_node.is_finished_ok


if __name__ == '__main__':
    _example_gaussian_spin(load_code("gaussian-16@tigu"))
