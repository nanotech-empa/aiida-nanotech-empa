import os
import ase.io



from aiida.orm import  StructureData, Int, List, Str
from aiida.orm import load_code
from aiida.engine import run_get_node
from aiida.plugins import WorkflowFactory

Cp2kReplicaWorkChain = WorkflowFactory('nanotech_empa.cp2k.replica_chain')

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
GEO_FILE = "c2h4.xyz"

def _example_cp2k_replicachain(cp2k_code,targets,restartpk):

    builder = Cp2kReplicaWorkChain.get_builder()

    builder.metadata.label = 'Cp2kReplicaWorkChain'
    builder.metadata.description = 'test description'
    builder.code = cp2k_code
    #builder.walltime_seconds = Int(600)
    ase_geom = ase.io.read(os.path.join(DATA_DIR, GEO_FILE))
    builder.structure = StructureData(ase=ase_geom)
    builder.max_nodes = Int(2)
    builder.constraints = Str(
        'fixed 1 , collective 1 [ev/angstrom^2] 40 [angstrom] 1.33 , collective 2 [ev/angstrom^2] 40 [angstrom] 1.09, collective 3 [ev/angstrom^2] 40 [angstrom] 1.87'
    )
    builder.colvars = Str('distance atoms 1 2 , distance atoms 1 3, distance atoms 5 6 ')

    builder.multiplicity = Int(0)
    builder.continuation_of = Int(restartpk)

    builder.protocol = Str('debug')
    builder.colvars_targets = List(list=targets)
    builder.colvars_increments = List(list=[0.06,0.05,0.0])

    _, calc_node = run_get_node(builder)

    assert calc_node.is_finished_ok

    #replicachain_out_dict = dict(calc_node.outputs.output_parameters)
    #print()
    #for k in replicachain_out_dict:
    #    print("  {}: {}".format(k, replicachain_out_dict[k]))
    return calc_node.pk


def example_cp2k_replicachain_rks(cp2k_code):
    pk = _example_cp2k_replicachain(cp2k_code,0)
    pk = _example_cp2k_replicachain(cp2k_code,pk)


#def example_cp2k_replicachain_rks_continuation(cp2k_code):
#    _example_cp2k_replicachain(cp2k_code,1):
#    _example_cp2k_slabopt(cp2k_code, 1)


if __name__ == '__main__':
    print("#### RKS")
    pk = _example_cp2k_replicachain(load_code("cp2k@localhost"),[1.40,1.21,1.87],0)
    print(f"#### RKS continuation from pk {pk}")
    pk = _example_cp2k_replicachain(load_code("cp2k@localhost"),[1.47,1.27,1.87],pk)

#    print("#### UKS")
#    _example_cp2k_replicachain(load_code("cp2k@localhost"), 1)
