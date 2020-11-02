from ase.io import read
from aiida.orm import Code, Float, Int, Str
from aiida.plugins import DataFactory, WorkflowFactory
from aiida.engine import run

# AiiDA classes.
StructureData = DataFactory('structure')
NanoribbonWorkChain = WorkflowFactory('nanotech_empa.nanoribbon')

builder = NanoribbonWorkChain.get_builder()

# Calculation settings.
builder.max_kpoints = Int(2)
builder.precision = Float(0.0)

# Resources.
builder.max_nodes = Int(1)
builder.mem_node = Int(32)

# Codes.
builder.pw_code = Code.get_from_string('pw@localhost')
builder.pp_code = Code.get_from_string('pp@localhost')
builder.projwfc_code = Code.get_from_string('projwfc@localhost')

# Inputs
builder.structure = StructureData(ase=read('c2h2_no_spin.xyz'))
builder.pseudo_family = Str('SSSP_modified')

# Metadata
builder.metadata = {
    "description": "Test calculation no spin",
    "label": "NanoribbonWorkChain",
}

run(builder)