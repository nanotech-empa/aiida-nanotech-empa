import os
import ase.io
from aiida import orm, engine, plugins

StructureData = plugins.DataFactory('structure')
Cp2kFragmentSeparationWorkChain = plugins.WorkflowFactory(
    'nanotech_empa.cp2k.fragment_separation')

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
GEO_FILE = "h2_on_hbn.xyz"


def _example_cp2k_ads_ene(cp2k_code, mult):
    """Example of running a workflow to compute the adsorption energy of a molecule on substrate."""

    builder = Cp2kFragmentSeparationWorkChain.get_builder()

    builder.metadata.label = 'Cp2kFragmentSeparationWorkChain'
    builder.metadata.description = 'test description'
    builder.code = cp2k_code
    ase_geom = ase.io.read(os.path.join(DATA_DIR, GEO_FILE))
    builder.structure = StructureData(ase=ase_geom)
    builder.fixed_atoms = orm.List(
        list=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])

    builder.fragments = {
        'molecule':
        orm.List(list=[0, 1]),
        'slab':
        orm.List(
            list=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]),
    }

    builder.multiplicities = {
        'all': orm.Int(mult),
        'molecule': orm.Int(mult),
        'slab': orm.Int(0),
    }

    mag = []
    if mult == 1:
        builder.uks = orm.Bool(True)
        mag = [0 for i in ase_geom]
        mag[0] = 1
        mag[1] = -1

    builder.magnetization_per_site = orm.List(list=mag)

    builder.options = {
        'all': {
            "max_wallclock_seconds": 1200,
            "resources": {
                "num_machines": 1,
                "num_mpiprocs_per_machine": 1,
            },
        },
        'molecule': {
            "max_wallclock_seconds": 1200,
            "resources": {
                "num_machines": 1,
                "num_mpiprocs_per_machine": 1,
            },
        },
        'slab': {
            "max_wallclock_seconds": 1200,
            "resources": {
                "num_machines": 1,
                "num_mpiprocs_per_machine": 1,
            },
        },
    }

    builder.protocol = orm.Str('debug')

    _, calc_node = engine.run_get_node(builder)

    assert calc_node.is_finished_ok

    adsorption_energy_out_dict = dict(calc_node.outputs.energies)

    for k in adsorption_energy_out_dict:
        print(f"{k}: {adsorption_energy_out_dict[k]}")


def example_cp2k_ads_ene_rks(cp2k_code):
    _example_cp2k_ads_ene(cp2k_code, 0)


def example_cp2k_slabopt_uks(cp2k_code):
    _example_cp2k_ads_ene(cp2k_code, 1)


if __name__ == '__main__':
    print("#### RKS")
    _example_cp2k_ads_ene(orm.load_code("cp2k@localhost"), 0)

    print("#### UKS")
    _example_cp2k_ads_ene(orm.load_code("cp2k@localhost"), 1)
