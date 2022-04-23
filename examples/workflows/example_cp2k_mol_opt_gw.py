import os

from ase import Atoms

from aiida.orm import StructureData, Bool, Str, Int, List
from aiida.orm import load_code
from aiida.engine import run_get_node
from aiida.plugins import WorkflowFactory

Cp2kMoleculeOptGwWorkChain = WorkflowFactory('nanotech_empa.cp2k.mol_opt_gw')

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
GEO_FILE = "h2.xyz"


def _example_cp2k_mol_opt_gw(cp2k_code, geo_opt):

    builder = Cp2kMoleculeOptGwWorkChain.get_builder()

    builder.metadata.description = 'H2 gas'
    builder.code = cp2k_code

    ase_geom = Atoms('HH',
                     positions=[[0, 0, 0], [0.75, 0, 0]],
                     cell=[4.0, 4.0, 4.0])
    mag_list = [-1, 1]

    builder.structure = StructureData(ase=ase_geom)
    builder.magnetization_per_site = List(list=mag_list)

    builder.protocol = Str('gpw_std')
    builder.multiplicity = Int(1)

    builder.geo_opt = Bool(False)
    if geo_opt:
        builder.geo_opt = Bool(True)

    builder.debug = Bool(True)
    builder.walltime_seconds = Int(5 * 60)
    builder.max_nodes = Int(1)

    _, calc_node = run_get_node(builder)

    assert calc_node.is_finished_ok

    gw_ic_res = dict(calc_node.outputs.gw_ic_parameters)
    print()
    for k in gw_ic_res:
        print("  {}: {}".format(k, gw_ic_res[k]))
    print()


def example_cp2k_mol_opt_gw_geo_opt(cp2k_code):
    _example_cp2k_mol_opt_gw(cp2k_code, geo_opt=True)


def example_cp2k_mol_opt_gw_no_geo_opt(cp2k_code):
    _example_cp2k_mol_opt_gw(cp2k_code, geo_opt=False)


if __name__ == '__main__':
    print("# geo opt #")
    example_cp2k_mol_opt_gw_geo_opt(load_code("cp2k@localhost"))
    print("# No geo opt #")
    example_cp2k_mol_opt_gw_no_geo_opt(load_code("cp2k@localhost"))
