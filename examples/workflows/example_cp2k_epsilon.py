import os
import ase.io
from aiida.engine import run_get_node
from aiida.orm import load_code
from aiida.plugins import DataFactory, WorkflowFactory

Cp2kDielectricWorkChain = WorkflowFactory("nanotech_empa.cp2k.epsilon")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
GEO_FILE = "si_bulk.xyz"
StructureData = DataFactory("core.structure")
Str = DataFactory("core.str")

def _example_cp2k_dielectic(cp2k_code):

    builder = Cp2kDielectricWorkChain.get_builder()
    builder.metadata.label = "Cp2kDielectricWorkChain"
    builder.metadata.description = "Si bulk dielectric constant"
    builder.cp2k_base.cp2k.code = cp2k_code
    builder.protocol = Str('debug')
    ase_geom = ase.io.read(os.path.join(DATA_DIR, GEO_FILE))
    builder.structure = StructureData(ase=ase_geom)

    builder.cp2k_base.cp2k.metadata.options = {
        "max_wallclock_seconds": 600,
        "resources": {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
            "num_cores_per_mpiproc": 1,
        },
    }

    _, calc_node = run_get_node(builder)

    epsilon_dict = calc_node.outputs.output.get_dict()

    print()
    for k in epsilon_dict:
        print(f"  {k}: {epsilon_dict[k]}")


if __name__ == "__main__":
    _example_cp2k_dielectic(load_code("cp2k@localhost"))
   