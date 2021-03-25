import numpy as np
from ase import Atoms

from aiida.orm import StructureData, Bool, Int, List
from aiida.orm import load_code
from aiida.engine import run_get_node
from aiida.plugins import WorkflowFactory

Cp2kMoleculeOptWorkChain = WorkflowFactory('nanotech_empa.cp2k_molecule_opt')

for mult in [0, 1]:

    print("####################################")
    print("####  mult={}".format(mult))
    print("####################################")

    builder = Cp2kMoleculeOptWorkChain.get_builder()

    builder.metadata.label = 'Cp2kMoleculeOptWorkChain'
    builder.metadata.description = 'test description'
    builder.code = load_code("cp2k-8.1@localhost")
    builder.walltime_seconds = Int(500)
    ase_geom = Atoms('HH', positions=[[0, 0, 0], [0.7, 0, 0]])
    ase_geom.cell = np.diag([4.0, 4.0, 4.0])
    builder.structure = StructureData(ase=ase_geom)

    builder.multiplicity = Int(mult)
    if mult == 1:
        builder.magnetization_per_site = List(list=[-1, 1])

    builder.debug = Bool(True)

    res, calc_node = run_get_node(builder)

    molopt_out_dict = dict(calc_node.outputs.output_parameters)
    for k in molopt_out_dict:
        print("{}: {}".format(k, molopt_out_dict[k]))