import os

import ase.io
from ase import Atoms

from aiida.orm import StructureData, Bool, Str, Int, List, Float, Dict
from aiida.orm import load_code
from aiida.engine import run_get_node
from aiida.plugins import WorkflowFactory

Cp2kAdsorbedGwIcWorkChain = WorkflowFactory('nanotech_empa.cp2k.ads_gw_ic')

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
GEO_FILE = "h2_on_au111.xyz"


def _example_cp2k_ads_gw_ic(cp2k_code, slab_included):

    builder = Cp2kAdsorbedGwIcWorkChain.get_builder()

    builder.metadata.description = os.path.splitext(GEO_FILE)[0]
    builder.code = cp2k_code

    if slab_included:
        ase_geom = ase.io.read(os.path.join(DATA_DIR, GEO_FILE))
        # Convert ase tags to magnetization list
        mag_list = [
            -1 if t == 1 else 1 if t == 2 else 0 for t in ase_geom.get_tags()
        ]
    else:
        ase_geom = Atoms('HH',
                         positions=[[0, 0, 0], [0.75, 0, 0]],
                         cell=[4.0, 4.0, 4.0])
        mag_list = [-1, 1]
        builder.ads_height = Float(3.0)

    builder.structure = StructureData(ase=ase_geom)
    builder.magnetization_per_site = List(list=mag_list)

    builder.protocol = Str('gpw_std')
    builder.multiplicity = Int(1)

    builder.geometry_mode = Str('ads_geo')

    builder.debug = Bool(True)
    builder.walltime_seconds = Int(5 * 60)
    builder.resources_scf = Dict(
        dict={
            'num_machines': 1,
            'num_mpiprocs_per_machine': 1,
            'num_cores_per_mpiproc': 1
        })
    builder.resources_gw = Dict(
        dict={
            'num_machines': 1,
            'num_mpiprocs_per_machine': 1,
            'num_cores_per_mpiproc': 1
        })
    builder.resources_ic = Dict(
        dict={
            'num_machines': 1,
            'num_mpiprocs_per_machine': 1,
            'num_cores_per_mpiproc': 1
        })

    _, calc_node = run_get_node(builder)

    assert calc_node.is_finished_ok

    gw_ic_res = dict(calc_node.outputs.gw_ic_parameters)
    print()
    for k in gw_ic_res:
        print("  {}: {}".format(k, gw_ic_res[k]))
    print()


def example_cp2k_ads_gw_ic_explicit_slab(cp2k_code):
    _example_cp2k_ads_gw_ic(cp2k_code, slab_included=True)


def example_cp2k_ads_gw_ic_implicit_slab(cp2k_code):
    _example_cp2k_ads_gw_ic(cp2k_code, slab_included=False)


if __name__ == '__main__':
    print("# Slab in geometry explicitly #")
    _example_cp2k_ads_gw_ic(load_code("cp2k@localhost"), slab_included=True)
    print("# Slab specified implicitly #")
    _example_cp2k_ads_gw_ic(load_code("cp2k@localhost"), slab_included=False)
