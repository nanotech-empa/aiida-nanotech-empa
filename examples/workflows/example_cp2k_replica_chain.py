import os
import ase.io



from aiida.orm import  StructureData, Int, List, Str
from aiida.orm import load_code
from aiida.engine import run_get_node
from aiida.plugins import WorkflowFactory

Cp2kReplicaWorkChain = WorkflowFactory('nanotech_empa.cp2k.replica_chain')

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
GEO_FILE = "c2h4.xyz"

def _example_cp2k_replicachain(cp2k_code):

    builder = Cp2kReplicaWorkChain.get_builder()

    builder.metadata.label = 'Cp2kReplicaWorkChain'
    builder.metadata.description = 'test description'
    builder.code = cp2k_code
    #builder.walltime_seconds = Int(600)
    ase_geom = ase.io.read(os.path.join(DATA_DIR, GEO_FILE))
    builder.structure = StructureData(ase=ase_geom)
    builder.max_nodes = Int(2)
    builder.constraints = Str(
        'fixed 1 , collective 1 [ev/angstrom^2] 40 [angstrom] 1.33 , collective 2 [ev/angstrom^2] 40 [angstrom] 1.09'
    )
    builder.colvars = Str('distance atoms 1 2 , distance atoms 2 5 ')

    builder.multiplicity = Int(0)

    builder.protocol = Str('debug')
    builder.colvars_targets = List(list=[1.45,1.29])
    builder.colvars_increments = List(list=[0.06,0.05])

    _, calc_node = run_get_node(builder)

    assert calc_node.is_finished_ok

    replicachain_out_dict = dict(calc_node.outputs.output_parameters)
    print()
    for k in replicachain_out_dict:
        print("  {}: {}".format(k, replicachain_out_dict[k]))


def example_cp2k_replicachain_rks(cp2k_code):
    _example_cp2k_replicachain(cp2k_code)


#def example_cp2k_slabopt_uks(cp2k_code):
#    _example_cp2k_slabopt(cp2k_code, 1)


if __name__ == '__main__':
    print("#### RKS")
    _example_cp2k_replicachain(load_code("cp2k@localhost"))

#    print("#### UKS")
#    _example_cp2k_replicachain(load_code("cp2k@localhost"), 1)
