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
from aiida_nanotech_empa.utils.cp2k_utils import get_kinds_section, tags_and_magnetization, dict_merge

Cp2kBaseWorkChain = WorkflowFactory('cp2k.base')





class Cp2kMoleculeOptWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("code", valid_type=Code)       
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
            "vdw", # +1 means one electron removed
            valid_type=Bool,
            default=lambda:Bool(False),
            required=False
        )       
        spec.input(
            "walltime_seconds", 
            valid_type=Int,
            default=lambda:Int(300),
            required=False
        )          
        spec.input("structure", valid_type=StructureData)
       
        #workchain outline
        spec.outline(
            cls.setup,
            cls.submit_calc,
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

    def submit_calc(self):
        
        #load input template
        with open(pathlib.Path(__file__).parent / '../../files/cp2k/molecule_opt_protocol.yml') as handle:
            protocols = yaml.safe_load(handle)
            input_dict = copy.deepcopy(protocols['default'])
            
        #get initial magnetization    
        structure, magnetization_tags = tags_and_magnetization(self.inputs.structure, self.inputs.magnetization_per_site)
        
        #make sure cell is big enough for MT poisson solver
        atoms = structure.get_ase()
        atoms.cell = 2*(np.ptp(atoms.positions, axis=0) + 5)
        atoms.center()
        
        builder = Cp2kBaseWorkChain.get_builder()
        builder.cp2k.code = self.inputs.code
        builder.cp2k.structure = StructureData(ase=atoms)
        builder.cp2k.file = {
            'basis': SinglefileData(file=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..", "files/cp2k", "BASIS_MOLOPT")),
            'pseudo': SinglefileData(file=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..", "files/cp2k", "POTENTIAL")),
        }

        
        # vdw
        if not self.inputs.vdw.value:
            input_dict['FORCE_EVAL']['DFT']['XC'].pop('VDW_POTENTIAL')

        #UKS
        if  self.inputs.multiplicity.value>0:
            input_dict['FORCE_EVAL']['DFT']['UKS'] = '.TRUE.'
            input_dict['FORCE_EVAL']['DFT']['MULTIPLICITY'] = self.inputs.multiplicity.value
        # KINDS section
        dict_merge(input_dict, get_kinds_section(structure, magnetization_tags))
        
        #walltime
        input_dict['GLOBAL']['WALLTIME'] = self.inputs.walltime_seconds.value
        #computational resources
        #nodes,tasks,threads = get_nodes(atoms=atoms,computer=self.inputs.code.computer)

        builder.cp2k.metadata.options.resources = {
            'num_machines': 1,
            'num_mpiprocs_per_machine': 1,
        }
        builder.cp2k.metadata.options.max_wallclock_seconds = self.inputs.walltime_seconds.value
        
        #parser
        builder.cp2k.metadata.options.parser_name = "cp2k_advanced_parser"
        
        #handlers
        builder.handler_overrides = Dict(dict={'resubmit_unconverged_geometry':True})
        
        #cp2k input dictionary
        builder.cp2k.parameters = Dict(dict=input_dict)
        
        submitted_node = self.submit(builder)
        return ToContext(opt=submitted_node)

    def finalize(self): 
        self.report("Finalizing...")
        self.out('opt_structure',self.ctx.opt.outputs.output_structure)
