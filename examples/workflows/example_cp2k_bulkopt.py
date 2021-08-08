import os
import ase.io

from aiida.orm import StructureData, Bool, Int, List, Str
from aiida.orm import load_code
from aiida.engine import run_get_node
from aiida.plugins import WorkflowFactory

Cp2kCellOptWorkChain = WorkflowFactory('nanotech_empa.cp2k.cell_opt')

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
GEO_FILE = "si_bulk.xyz"


def _example_cp2k_cellopt(cp2k_code, cell_opt, mult):

    builder = Cp2kCellOptWorkChain.get_builder()

    builder.metadata.label = 'Cp2kBulkOptWorkChain'
    builder.metadata.description = 'test description'
    builder.code = cp2k_code
    builder.walltime_seconds = Int(600)
    ase_geom = ase.io.read(os.path.join(DATA_DIR, GEO_FILE))
    builder.structure = StructureData(ase=ase_geom)
    builder.max_nodes = Int(1)
    if cell_opt:
        builder.symmetry = Str('ORTHORHOMBIC')
        builder.cell_freedom = Str('KEEP_SYMMETRY')

    builder.multiplicity = Int(mult)
    mag = [0 for i in ase_geom]
    if mult == 1:
        mag[0] = 1
        mag[1] = -1
        builder.magnetization_per_site = List(list=mag)

    builder.debug = Bool(True)

    _, calc_node = run_get_node(builder)

    assert calc_node.is_finished_ok

    cellopt_out_dict = dict(calc_node.outputs.output_parameters)
    print()
    for k in cellopt_out_dict:
        print("  {}: {}".format(k, cellopt_out_dict[k]))


def example_cp2k_bulkopt_rks(cp2k_code):
    _example_cp2k_cellopt(cp2k_code, False, 0)


def example_cp2k_cellopt_rks(cp2k_code):
    _example_cp2k_cellopt(cp2k_code, True, 0)


def example_cp2k_cellopt_uks(cp2k_code):
    _example_cp2k_cellopt(cp2k_code, True, 1)


if __name__ == '__main__':
    print("#### Bulk opt  RKS")
    _example_cp2k_cellopt(load_code("cp2k@localhost"), False, 0)

    print("#### Cell opt RKS")
    _example_cp2k_cellopt(load_code("cp2k@localhost"), True, 0)

    print("#### Cell opt UKS")
    _example_cp2k_cellopt(load_code("cp2k@localhost"), True, 1)
