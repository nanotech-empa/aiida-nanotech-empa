import os

from ase.io import read
from aiida.orm import Float, Int, Str
from aiida.orm import load_code
from aiida.plugins import DataFactory, WorkflowFactory
from aiida.engine import run_get_node

from aiida_nanotech_empa.utils.cube_utils import cube_from_qe_pp_arraydata

# AiiDA classes.
StructureData = DataFactory('structure')
NanoribbonWorkChain = WorkflowFactory('nanotech_empa.nanoribbon')

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def _example_nanoribbon(qe_pw_code, qe_pp_code, qe_projwfc_code, geo_file,
                        description):

    builder = NanoribbonWorkChain.get_builder()

    # Calculation settings.
    builder.max_kpoints = Int(2)
    builder.precision = Float(0.0)

    # Resources.
    builder.max_nodes = Int(1)
    builder.mem_node = Int(32)

    # Codes.
    builder.pw_code = qe_pw_code
    builder.pp_code = qe_pp_code
    builder.projwfc_code = qe_projwfc_code

    # Inputs
    builder.structure = StructureData(ase=read(geo_file))
    builder.pseudo_family = Str("SSSP_modified")

    # Metadata
    builder.metadata = {
        "description": description,
        "label": "NanoribbonWorkChain",
    }

    _, node = run_get_node(builder)

    assert node.is_finished_ok

    if 'spin_density_arraydata' in node.outputs:
        cube = cube_from_qe_pp_arraydata(node.outputs.spin_density_arraydata)
        cube.write_cube_file(os.path.join(OUTPUT_DIR, "spin.cube"))
        print("Wrote spin.cube!")


def example_nanoribbon_no_spin(qe_pw_code, qe_pp_code, qe_projwfc_code):
    _example_nanoribbon(qe_pw_code, qe_pp_code, qe_projwfc_code,
                        os.path.join(DATA_DIR, 'c2h2_no_spin.xyz'),
                        "Test calculation no spin")


def example_nanoribbon_spin(qe_pw_code, qe_pp_code, qe_projwfc_code):
    _example_nanoribbon(qe_pw_code, qe_pp_code, qe_projwfc_code,
                        os.path.join(DATA_DIR, 'c2h2_spin.xyz'),
                        "Test calculation spin")


if __name__ == '__main__':
    example_nanoribbon_no_spin(load_code('pw@localhost'),
                               load_code('pp@localhost'),
                               load_code('projwfc@localhost'))

    example_nanoribbon_spin(load_code('pw@localhost'),
                            load_code('pp@localhost'),
                            load_code('projwfc@localhost'))
