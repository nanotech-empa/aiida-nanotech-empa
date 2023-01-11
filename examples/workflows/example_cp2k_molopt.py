from aiida.engine import run_get_node
from aiida.orm import Int, List, Str, StructureData, load_code
from aiida.plugins import WorkflowFactory
from ase import Atoms

Cp2kMoleculeOptWorkChain = WorkflowFactory('nanotech_empa.cp2k.molecule_opt')


def _example_cp2k_molopt(cp2k_code, mult):

    builder = Cp2kMoleculeOptWorkChain.get_builder()

    builder.metadata.label = 'Cp2kMoleculeOptWorkChain'
    builder.metadata.description = 'test description'
    builder.code = cp2k_code
    ase_geom = Atoms('HH',
                     positions=[[0, 0, 0], [0.75, 0, 0]],
                     cell=[4.0, 4.0, 4.0])
    builder.structure = StructureData(ase=ase_geom)

    builder.multiplicity = Int(mult)
    if mult == 1:
        builder.magnetization_per_site = List([-1, 1])

    builder.options = {
        "max_wallclock_seconds": 600,
        "resources": {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
            "num_cores_per_mpiproc": 1,
        },
    }
    builder.protocol = Str('debug')

    _, calc_node = run_get_node(builder)

    assert calc_node.is_finished_ok

    molopt_out_dict = dict(calc_node.outputs.output_parameters)
    print()
    for k in molopt_out_dict:
        print(f"  {k}: {molopt_out_dict[k]}")


def example_cp2k_molopt_rks(cp2k_code):
    _example_cp2k_molopt(cp2k_code, 0)


def example_cp2k_molopt_uks(cp2k_code):
    _example_cp2k_molopt(cp2k_code, 1)


if __name__ == '__main__':
    for mult in [0, 1]:
        print()
        print("####################################")
        print(f"####  mult={mult}")
        print("####################################")
        _example_cp2k_molopt(load_code("cp2k@localhost"), mult)
