import os

import ase.io

from aiida.orm import StructureData, Bool, Str, Int, List
from aiida.orm import load_code
from aiida.engine import run_get_node
from aiida.plugins import WorkflowFactory

Cp2kAdsorbedGwIcWorkChain = WorkflowFactory('nanotech_empa.cp2k.ads_gw_ic')

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
GEO_FILE = "h2_on_au111.xyz"


def example_cp2k_ads_gw_ic(cp2k_code):

    ase_geom = ase.io.read(os.path.join(DATA_DIR, GEO_FILE))
    # Convert ase tags to magnetization list with the following mapping:
    # tag 1 -> -1 mag
    # tag 2 -> +1 mag
    # everything else: 0 mag
    mag_list = [
        -1 if t == 1 else 1 if t == 2 else 0 for t in ase_geom.get_tags()
    ]

    builder = Cp2kAdsorbedGwIcWorkChain.get_builder()

    builder.metadata.description = os.path.splitext(GEO_FILE)[0]
    builder.code = cp2k_code

    builder.ads_struct = StructureData(ase=ase_geom)
    builder.magnetization_per_site = List(list=mag_list)

    builder.protocol = Str('gpw_std')
    builder.multiplicity = Int(1)

    builder.geometry_mode = Str('ads_geo')

    builder.debug = Bool(True)
    builder.walltime_seconds = Int(5 * 60)
    builder.max_nodes = Int(1)

    _, calc_node = run_get_node(builder)

    assert calc_node.is_finished_ok

    gw_ic_res = dict(calc_node.outputs.gw_ic_parameters)
    print()
    for k in gw_ic_res:
        print("  {}: {}".format(k, gw_ic_res[k]))


if __name__ == '__main__':
    example_cp2k_ads_gw_ic(load_code("cp2k@localhost"))
