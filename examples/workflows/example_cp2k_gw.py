import numpy as np
from ase import Atoms

from aiida.orm import StructureData, Bool, Str, Int, Float, List
from aiida.orm import load_code
from aiida.engine import run_get_node
from aiida.plugins import WorkflowFactory

Cp2kMoleculeGwWorkChain = WorkflowFactory('nanotech_empa.cp2k_molecule_gw')

for image_charge in [False, True]:
    for protocol in ['gpw_std', 'gapw_std', 'gapw_hq']:
        for mult in [0, 1]:

            print("####################################")
            print("#### ic={}; {}; mult={}".format(image_charge, protocol,
                                                   mult))
            print("####################################")

            builder = Cp2kMoleculeGwWorkChain.get_builder()

            builder.metadata.label = 'Cp2kMoleculeGwWorkChain'
            builder.metadata.description = 'test description'
            builder.code = load_code("cp2k@localhost")

            ase_geom = Atoms('HH', positions=[[0, 0, 0], [0.7, 0, 0]])
            ase_geom.cell = np.diag([4.0, 4.0, 4.0])
            builder.structure = StructureData(ase=ase_geom)

            builder.protocol = Str(protocol)

            builder.multiplicity = Int(mult)
            if mult == 1:
                builder.magnetization_per_site = List(list=[-1, 1])

            builder.debug = Bool(True)

            builder.image_charge = Bool(image_charge)
            builder.z_ic_plane = Float(0.8)

            res, calc_node = run_get_node(builder)

            gw_out_dict = dict(calc_node.outputs.gw_output_parameters)
            for k in gw_out_dict:
                print("{}: {}".format(k, gw_out_dict[k]))
