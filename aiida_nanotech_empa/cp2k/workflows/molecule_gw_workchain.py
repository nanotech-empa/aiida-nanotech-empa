import os
import sys
import pathlib
import yaml
import copy
import numpy as np


from aiida.engine import WorkChain, ToContext, if_
from aiida.orm import Int, Float, Str, Bool, Code, Dict, List,SinglefileData
from aiida.orm import SinglefileData, StructureData, RemoteData
from aiida.plugins import CalculationFactory, WorkflowFactory
from aiida_nanotech_empa.cp2k.utils.utils import get_kinds_section_gw, tags_and_magnetization, dict_merge
from aiida_cp2k.calculations import Cp2kCalculation


PROTOCOLS = {
    'gw' : ['gw_first_step','gw_second_step'],
    'gw_ic' : ['gw_first_step','gw_ic_second_step'],
    'gw_hq' : ['gw_hq_first_step','gw_hq_second_step'],
    'gw_hq_ic' : ['gw_hq_first_step','gw_ic_hq_second_step']    
}


class Cp2kMoleculeGwWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("code", valid_type=Code)  
        spec.input(
            "gw_type", 
            valid_type=Str,
            default=lambda:Str('gw'),
            required=False
        )  
        spec.input(
            "z_ic_plane", 
            valid_type=Float,
            default=lambda:Float(8.22),
            required=False
        )                    
        spec.input(
            "charge", # +1 means one electron removed
            valid_type=Int,
            default=lambda:Int(0),
            required=False
        )        
        spec.input(
            "multiplicity", 
            valid_type=Int,
            default=lambda:Int(0),
            required=False
        )  
        spec.input(
            "magnetization_per_site", 
            valid_type=List,
            default=lambda:List(list=[]),
            required=False
        )       
        spec.input(
            "walltime_seconds", 
            valid_type=Int,
            default=lambda:Int(300),
            required=False
        )          
        spec.input("structure", valid_type=StructureData)
        
        spec.outline(
            cls.setup,
            cls.submit_first_step,
            cls.submit_second_step,
            cls.finalize
        )
        spec.outputs.dynamic = True

        spec.exit_code(
            390,
            "ERROR_TERMINATION",
            message="One or more steps of the work chain failed.",
        )

    def setup(self):
        self.report("Inspecting input and setting up things")

        # --------------------------------------------------

    def submit_first_step(self):
        
        #load input template
        with open(pathlib.Path(__file__).parent / '../files/gw_protocols.yml') as handle:
            protocols = yaml.safe_load(handle)
            input_dict = copy.deepcopy(protocols[PROTOCOLS[self.inputs.gw_type.value][0]])

        structure = self.inputs.structure
        magnetization_per_site = copy.deepcopy(self.inputs.magnetization_per_site)
        #add ghost atoms in case of gw-ic
        if 'ic' in self.inputs.gw_type.value:
            atoms = self.inputs.structure.get_ase()
            image = atoms.copy()
            image.set_masses([int(999) for a in range(len(image))])
            image.positions[:, 2] = 2*self.inputs.z_ic_plane.value - atoms.positions[:, 2]
            magnetization_per_site += [int(0) for i in range(len(image))]
            structure = StructureData(ase=atoms+image)

        #get initial magnetization           
        structure, magnetization_tags = tags_and_magnetization(structure, magnetization_per_site)
        
        #make sure cell is big enough for MT poisson solver
        atoms = structure.get_ase()
        atoms.cell = 2*(np.ptp(atoms.positions, axis=0) + 5)
        atoms.center()
        
        builder = Cp2kCalculation.get_builder()
        builder.code = self.inputs.code
        builder.structure = StructureData(ase=atoms)
        builder.file = {
            'basis': SinglefileData(file=os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "files", "GW_BASIS_SET")),
            'pseudo': SinglefileData(file=os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "files", "ALL_POTENTIALS")),
        }

        #UKS
        if  self.inputs.multiplicity.value>0:
            input_dict['FORCE_EVAL']['DFT']['UKS'] = '.TRUE.'     
            input_dict['FORCE_EVAL']['DFT']['MULTIPLICITY'] = self.inputs.multiplicity.value   
        # KINDS section
        dict_merge(input_dict, get_kinds_section_gw(structure, magnetization_tags))
        
        #computational resources
        #nodes,tasks,threads = get_nodes(atoms=atoms,computer=self.inputs.code.computer)
        builder.parameters = Dict(dict=input_dict)
        builder.metadata.options.resources = {
            'num_machines': 1,
            'num_mpiprocs_per_machine': 1,
        }
        builder.metadata.options.max_wallclock_seconds = self.inputs.walltime_seconds.value
        
        #parser
        builder.metadata.options.parser_name = "cp2k_advanced_parser"
        
        #handlers
        
        #description section
        
        submitted_node = self.submit(builder)
        return ToContext(first_step=submitted_node)

    def submit_second_step(self):
        
        #load input template
        with open(pathlib.Path(__file__).parent / '../files/gw_protocols.yml') as handle:
            protocols = yaml.safe_load(handle)
            input_dict = copy.deepcopy(protocols[PROTOCOLS[self.inputs.gw_type.value][1]])

        structure = self.ctx.first_step.inputs.structure           
        magnetization_per_site = copy.deepcopy(self.inputs.magnetization_per_site)
        if len(magnetization_per_site) < len(structure.get_ase()):
               magnetization_per_site += [int(0) for i in range(len(magnetization_per_site))]
        #get initial magnetization    
        structure, magnetization_tags = tags_and_magnetization(structure, magnetization_per_site)
        
        builder = Cp2kCalculation.get_builder()
        #restart from wfn of step1
        builder.parent_calc_folder = self.ctx.first_step.outputs.remote_folder
        builder.code = self.inputs.code
        builder.structure = structure
        builder.file = {
            'basis': SinglefileData(file=os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "files", "GW_BASIS_SET")),
            'pseudo': SinglefileData(file=os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "files", "ALL_POTENTIALS")),
        }

        #UKS
        if  self.inputs.multiplicity.value>0:
            input_dict['FORCE_EVAL']['DFT']['UKS'] = '.TRUE.'  
            input_dict['FORCE_EVAL']['DFT']['MULTIPLICITY'] = self.inputs.multiplicity.value     
        # KINDS section
        dict_merge(input_dict, get_kinds_section_gw(structure, magnetization_tags))
        
        #computational resources
        #nodes,tasks,threads = get_nodes(atoms=atoms,computer=self.inputs.code.computer)
        builder.parameters = Dict(dict=input_dict)
        builder.metadata.options.resources = {
            'num_machines': 1,
            'num_mpiprocs_per_machine': 1,
        }
        builder.metadata.options.max_wallclock_seconds = self.inputs.walltime_seconds.value
        
        #parser
        builder.metadata.options.parser_name = "cp2k_advanced_parser"
        
        #handlers
        
        #description section
        
        submitted_node = self.submit(builder)
        return ToContext(second_step=submitted_node)        

    def finalize(self): 
        self.report("Finalizing...")
        #self.out('opt_structure',self.ctx.opt.outputs.output_structure)
