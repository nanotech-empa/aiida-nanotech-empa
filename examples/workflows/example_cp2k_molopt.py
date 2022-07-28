from ase import Atoms

from aiida.orm import StructureData, Bool, Int, List, Str
from aiida.orm import load_code
from aiida.engine import run_get_node
from aiida.plugins import WorkflowFactory

Cp2kMoleculeOptWorkChain = WorkflowFactory('nanotech_empa.cp2k.molecule_opt')


def _example_cp2k_molopt(cp2k_code, mult):

    builder = Cp2kMoleculeOptWorkChain.get_builder()

    builder.metadata.label = 'Cp2kMoleculeOptWorkChain'
    builder.metadata.description = 'test description'
    builder.code = cp2k_code
    builder.walltime_seconds = Int(600)
    ase_geom = Atoms('HH',
                     positions=[[0, 0, 0], [0.75, 0, 0]],
                     cell=[4.0, 4.0, 4.0])
    builder.structure = StructureData(ase=ase_geom)

    builder.constraints = Str(
        'fixed xy 1, collective 1 [ev/angstrom^2] 40 [angstrom] 0.8')
    builder.colvars = Str('distance atoms 1 2 ')

    builder.multiplicity = Int(mult)
    if mult == 1:
        builder.magnetization_per_site = List(list=[-1, 1])

    builder.debug = Bool(True)

    _, calc_node = run_get_node(builder)

    assert calc_node.is_finished_ok

    molopt_out_dict = dict(calc_node.outputs.output_parameters)
    print()
    for k in molopt_out_dict:
        print("  {}: {}".format(k, molopt_out_dict[k]))


def example_cp2k_molopt_rks(cp2k_code):
    _example_cp2k_molopt(cp2k_code, 0)


def example_cp2k_molopt_uks(cp2k_code):
    _example_cp2k_molopt(cp2k_code, 1)


if __name__ == '__main__':
    for mult in [0, 1]:
        print()
        print("####################################")
        print("####  mult={}".format(mult))
        print("####################################")
        _example_cp2k_molopt(load_code("cp2k@localhost"), mult)
