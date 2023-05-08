import os

import ase
from aiida import engine, orm, plugins

Cp2kAdsorbedGwIcWorkChain = plugins.WorkflowFactory("nanotech_empa.cp2k.ads_gw_ic")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
GEO_FILE = "h2_on_au111.xyz"


def _example_cp2k_ads_gw_ic(cp2k_code, slab_included):
    builder = Cp2kAdsorbedGwIcWorkChain.get_builder()

    builder.metadata.description = os.path.splitext(GEO_FILE)[0]
    builder.code = cp2k_code

    if slab_included:
        ase_geom = ase.io.read(os.path.join(DATA_DIR, GEO_FILE))
        # Convert ase tags to magnetization list
        mag_list = [-1 if t == 1 else 1 if t == 2 else 0 for t in ase_geom.get_tags()]
    else:
        ase_geom = ase.Atoms(
            "HH", positions=[[0, 0, 0], [0.75, 0, 0]], cell=[4.0, 4.0, 4.0]
        )
        mag_list = [-1, 1]
        builder.ads_height = orm.Float(3.0)

    builder.structure = orm.StructureData(ase=ase_geom)
    builder.magnetization_per_site = orm.List(mag_list)

    builder.protocol = orm.Str("gpw_std")
    builder.multiplicity = orm.Int(1)

    builder.geometry_mode = orm.Str("ads_geo")

    builder.debug = orm.Bool(True)
    builder.options.scf = {
        "max_wallclock_seconds": 600,
        "resources": {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
            "num_cores_per_mpiproc": 1,
        },
    }
    builder.options.gw = {
        "max_wallclock_seconds": 600,
        "resources": {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
            "num_cores_per_mpiproc": 1,
        },
    }
    builder.options.ic = {
        "max_wallclock_seconds": 600,
        "resources": {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
            "num_cores_per_mpiproc": 1,
        },
    }
    _, calc_node = engine.run_get_node(builder)

    assert calc_node.is_finished_ok

    gw_ic_res = dict(calc_node.outputs.gw_ic_parameters)
    print()
    for k in gw_ic_res:
        print(f"  {k}: {gw_ic_res[k]}")
    print()


def example_cp2k_ads_gw_ic_explicit_slab(cp2k_code):
    _example_cp2k_ads_gw_ic(cp2k_code, slab_included=True)


def example_cp2k_ads_gw_ic_implicit_slab(cp2k_code):
    _example_cp2k_ads_gw_ic(cp2k_code, slab_included=False)


if __name__ == "__main__":
    print("# Slab in geometry explicitly #")
    _example_cp2k_ads_gw_ic(orm.load_code("cp2k@localhost"), slab_included=True)
    print("# Slab specified implicitly #")
    _example_cp2k_ads_gw_ic(orm.load_code("cp2k@localhost"), slab_included=False)
