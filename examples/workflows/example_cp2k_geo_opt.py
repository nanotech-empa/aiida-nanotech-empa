import os

import ase.io
from aiida.engine import run_get_node
from aiida.orm import Int, List, Str, StructureData, load_code
from aiida.plugins import WorkflowFactory

Cp2kGeoOptWorkChain = WorkflowFactory("nanotech_empa.cp2k.geo_opt")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
GEO_FILE = "c2h2_on_au111.xyz"


def _example_cp2k_geo_opt(cp2k_code,sys_type,uks):

    builder = Cp2kGeoOptWorkChain.get_builder()

    builder.metadata.label = "Cp2kGeoOptWorkChain"
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
                }
                }
    magnetization_per_site = [0 for i in range(len(ase_geom))]

    if sys_type == 'SlabXY':
        builder.dft_params = Dict(
            {"protocol":"debug", 
            "uks": uks,  
            "periodic": 'XYZ',
            "vdw": False,
            "cutoff": 300,
            }
        )
        if uks:
            magnetization_per_site[1]=1
            magnetization_per_site[2]=-1
            builder.dft_params = Dict(
                {"protocol":"debug",
                "uks": uks, 
                "magnetization_per_site": magnetization_per_site,
                "charge" : 0,
                "periodic": 'XYZ',
                "multiplicity":1,
                "cutoff": 300,
                }
            )

        builder.sys_params=Dict(
            {"cell_opt": False,
            "constraints":'fixed 5..40 , collective 1 [ev/angstrom^2] 40 [angstrom] 1.36 , collective 2 [ev/angstrom^2] 40 [angstrom] 1.07',
            "colvars":'distance atoms 2 3 , distance atoms 1 2',
            }
        )

    _, calc_node = run_get_node(builder)

    assert calc_node.is_finished_ok

    slabopt_out_dict = dict(calc_node.outputs.output_parameters)
    print()
    for k in slabopt_out_dict:
        print(f"  {k}: {slabopt_out_dict[k]}")


def example_cp2k_slab_opt_rks(cp2k_code):
    _example_cp2k_geo_opt(cp2k_code, 'SlabXY',False)


def example_cp2k_slab_opt_uks(cp2k_code):
    _example_cp2k_geo_opt(cp2k_code, 'SlabXY',True)


if __name__ == "__main__":
    print("#### Slab RKS")
    _example_cp2k_geo_opt(load_code("cp2k@localhost"), 'SlabXY',False)

    print("#### Slab UKS")
    _example_cp2k_geo_opt(load_code("cp2k@localhost"), 'SlabXY', True)
