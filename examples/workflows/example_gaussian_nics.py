import os
import click
from ase.build import molecule
import numpy as np
from aiida import engine, orm
from aiida.plugins import WorkflowFactory

import aiida_nanotech_empa.utils.gaussian_wcs_postprocess as pp

GaussianNicsWorkChain = WorkflowFactory("nanotech_empa.gaussian.nics")

def _example_gaussian_nics(gaussian_code,opt):
    ase_geom = molecule('C6H6')
    ase_geom.cell = np.diag([10.0, 10.0, 10.0])

    builder = GaussianNicsWorkChain.get_builder()
    builder.gaussian_code = gaussian_code
    builder.structure = orm.StructureData(ase=ase_geom)
    builder.functional = orm.Str("B3LYP")
    builder.basis_set = orm.Str("6-311G(d,p)")
    builder.opt = orm.Bool(opt)
    builder.options = orm.Dict(
        {
            "resources": {
                "tot_num_mpiprocs": 4,
                "num_machines": 1,
            },
            "max_wallclock_seconds": 1 * 60 * 60,
            "max_memory_kb": 8 * 1024 * 1024, #GB
        }
    )    
    
    _, wc_node = engine.run_get_node(builder)
    
    assert wc_node.is_finished_ok
    return wc_node.pk

@click.command("cli")
@click.argument("gaussian_code", default="gaussian@localhost")
def run_nics(gaussian_code):
    #print("#### Running Gaussian NICS WorkChain with geo_opt ####")
    #uuid = _example_gaussian_nics(orm.load_code(gaussian_code),True)
    #print(f"WorkChain completed uuid: {uuid}")
    print("#### Running Gaussian NICS WorkChain without geo_opt ####")
    uuid = _example_gaussian_nics(orm.load_code(gaussian_code),False)
    print(f"WorkChain completed uuid: {uuid}")


if __name__ == "__main__":
    run_nics()