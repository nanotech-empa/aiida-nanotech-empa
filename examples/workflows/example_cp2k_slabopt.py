import os
import ase.io

from aiida.orm import StructureData, Int, List, Str
from aiida.orm import load_code
from aiida.engine import run_get_node
from aiida.plugins import WorkflowFactory

Cp2kSlabOptWorkChain = WorkflowFactory('nanotech_empa.cp2k.slab_opt')

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
GEO_FILE = "h2_on_hbn.xyz"


def _example_cp2k_slabopt(cp2k_code, mult):

    builder = Cp2kSlabOptWorkChain.get_builder()

    builder.metadata.label = 'Cp2kSlabOptWorkChain'
    builder.metadata.description = 'test description'
    builder.code = cp2k_code
    builder.walltime_seconds = Int(600)
    ase_geom = ase.io.read(os.path.join(DATA_DIR, GEO_FILE))
    builder.structure = StructureData(ase=ase_geom)
    builder.max_nodes = Int(1)
    builder.fixed_atoms = Str('3..18')

    builder.multiplicity = Int(mult)
    mag = [0 for i in ase_geom]
    if mult == 1:
        mag[0] = 1
        mag[1] = -1
        builder.magnetization_per_site = List(list=mag)

    builder.protocol = Str('debug')

    _, calc_node = run_get_node(builder)

    assert calc_node.is_finished_ok

    slabopt_out_dict = dict(calc_node.outputs.output_parameters)
    print()
    for k in slabopt_out_dict:
        print("  {}: {}".format(k, slabopt_out_dict[k]))


def example_cp2k_slabopt_rks(cp2k_code):
    _example_cp2k_slabopt(cp2k_code, 0)


def example_cp2k_slabopt_uks(cp2k_code):
    _example_cp2k_slabopt(cp2k_code, 1)


if __name__ == '__main__':
    print("#### RKS")
    _example_cp2k_slabopt(load_code("cp2k@localhost"), 0)

    print("#### UKS")
    _example_cp2k_slabopt(load_code("cp2k@localhost"), 1)
