import os

import ase.io
import numpy as np
from aiida.engine import run_get_node
from aiida.orm import Bool, Dict, List, Str, StructureData, load_code
from aiida.plugins import WorkflowFactory

GaussianCasscfSeriesWorkChain = WorkflowFactory("nanotech_empa.gaussian.casscf_series")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def _example_gaussian_casscf(gaussian_code, formchk_code, cubegen_code):
    ase_geom = ase.io.read(os.path.join(DATA_DIR, "benzene-diradical.xyz"))
    ase_geom.cell = np.diag([10.0, 10.0, 10.0])

    builder = GaussianCasscfSeriesWorkChain.get_builder()
    builder.gaussian_code = gaussian_code
    builder.formchk_code = formchk_code
    builder.cubegen_code = cubegen_code

    builder.structure = StructureData(ase=ase_geom)
    builder.init_functional = Str("UB3LYP")
    builder.basis_set = Str("STO-3G")

    builder.nm_list = List([(4, 4), (6, 6)])
    builder.multiplicity_list = List([1, 3])
    builder.mp2 = Bool(True)

    builder.options = Dict(
        {
            "resources": {
                "tot_num_mpiprocs": 1,
                "num_machines": 1,
            },
            "max_wallclock_seconds": 1 * 60 * 60,
            "max_memory_kb": 2 * 1024 * 1024,
        }
    )

    _, wc_node = run_get_node(builder)

    assert wc_node.is_finished_ok

    for nm in [(4, 4), (6, 6)]:
        m1_label = f"cas_{nm[0]}_{nm[1]}_m1_out_params"
        m3_label = f"cas_{nm[0]}_{nm[1]}_m3_out_params"

        casscf_ts = (
            wc_node.outputs[m3_label]["casscf_energy_ev"]
            - wc_node.outputs[m1_label]["casscf_energy_ev"]
        )

        casmp2_ts = (
            wc_node.outputs[m3_label]["casmp2_energy_ev"]
            - wc_node.outputs[m1_label]["casmp2_energy_ev"]
        )

        print(f"CAS({nm[0]}, {nm[1]})")
        print(f"    CASSCF T-S: {casscf_ts:.4f}")
        print(f"    CASMP2 T-S: {casmp2_ts:.4f}")


if __name__ == "__main__":
    _example_gaussian_casscf(
        load_code("gaussian@localhost"),
        load_code("formchk@localhost"),
        load_code("cubegen@localhost"),
    )
