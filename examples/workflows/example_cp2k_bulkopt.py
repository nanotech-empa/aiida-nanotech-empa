import os

import ase.io
from aiida.engine import run_get_node
from aiida.orm import Bool, Int, List, Str, StructureData, load_code
from aiida.plugins import WorkflowFactory

Cp2kBulkOptWorkChain = WorkflowFactory("nanotech_empa.cp2k.bulk_opt")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
GEO_FILE = "si_bulk.xyz"


def _example_cp2k_bulkopt(cp2k_code, cell_opt, mult):

    builder = Cp2kBulkOptWorkChain.get_builder()

    builder.metadata.label = "Cp2kBulkOptWorkChain"
    builder.metadata.description = "test description"
    builder.code = cp2k_code
    ase_geom = ase.io.read(os.path.join(DATA_DIR, GEO_FILE))
    builder.structure = StructureData(ase=ase_geom)
    builder.options = {
        "max_wallclock_seconds": 600,
        "resources": {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
            "num_cores_per_mpiproc": 1,
        },
    }
    builder.protocol = Str("debug")
    if cell_opt:
        builder.cell_opt = Bool(True)
        builder.symmetry = Str("ORTHORHOMBIC")
        builder.cell_freedom = Str("KEEP_SYMMETRY")
    builder.multiplicity = Int(mult)
    mag = [0 for i in ase_geom]
    if mult == 1:
        mag[0] = 1
        mag[1] = -1
        builder.magnetization_per_site = List(mag)

    _, calc_node = run_get_node(builder)

    assert calc_node.is_finished_ok

    bulkopt_out_dict = dict(calc_node.outputs.output_parameters)
    print()
    for k in bulkopt_out_dict:
        print(f"  {k}: {bulkopt_out_dict[k]}")


def example_cp2k_bulkopt_rks(cp2k_code):
    _example_cp2k_bulkopt(cp2k_code, False, 0)


def example_cp2k_cellopt_rks(cp2k_code):
    _example_cp2k_bulkopt(cp2k_code, True, 0)


def example_cp2k_cellopt_uks(cp2k_code):
    _example_cp2k_bulkopt(cp2k_code, True, 1)


if __name__ == "__main__":
    print("#### Bulk opt  RKS")
    _example_cp2k_bulkopt(load_code("cp2k@localhost"), False, 0)

    print("#### Cell opt RKS")
    _example_cp2k_bulkopt(load_code("cp2k@localhost"), True, 0)

    print("#### Cell opt UKS")
    _example_cp2k_bulkopt(load_code("cp2k@localhost"), True, 1)
