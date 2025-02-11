import os

import ase.io
import numpy as np
from aiida import engine, orm, plugins

GaussianRelaxWorkChain = plugins.WorkflowFactory("nanotech_empa.gaussian.relax")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def _example_gaussian_spin(gaussian_code):  # , formchk_code, cubegen_code):
    ase_geom = ase.io.read(os.path.join(DATA_DIR, "benzene.xyz"))
    ase_geom.cell = np.diag([10.0, 10.0, 10.0])

    builder = GaussianRelaxWorkChain.get_builder()
    builder.gaussian_code = gaussian_code
    # builder.formchk_code = formchk_code
    # builder.cubegen_code = cubegen_code
    builder.structure = orm.StructureData(ase=ase_geom)
    builder.functional = orm.Str("B3LYP")
    builder.empirical_dispersion = orm.Str("GD3")
    builder.basis_set = orm.Str("6-311+G(d,p)")
    # builder.basis_set_scf = orm.Str("STO-3G")
    builder.multiplicity = orm.Int(0)
    builder.tight = orm.Bool(True)
    builder.options = orm.Dict(
        {
            "resources": {
                "tot_num_mpiprocs": 4,
                "num_machines": 1,
            },
            "max_wallclock_seconds": 1 * 60 * 60,
            "max_memory_kb": 4 * 1024 * 1024,
        }
    )

    _, wc_node = engine.run_get_node(builder)

    assert wc_node.is_finished_ok

    # pp.make_report(wc_node, nb=False, save_image_loc=OUTPUT_DIR)


if __name__ == "__main__":
    _example_gaussian_spin(
        orm.load_code("gaussian@tigu"),
        # orm.load_code("formchk@localhost"),
        # orm.load_code("cubegen@localhost"),
    )
