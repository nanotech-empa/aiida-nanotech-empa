import copy
import os
import pathlib

import numpy as np
import yaml

from aiida.engine import ToContext, WorkChain,  while_
from aiida.orm import  Code, Dict, Str, StructureData

from aiida.plugins import CalculationFactory, WorkflowFactory

from aiida_nanotech_empa.utils import common_utils
from aiida_nanotech_empa.workflows.cp2k.cp2k_utils import (
    determine_kinds,
    dict_merge,
    get_cutoff,
    get_kinds_section,
)

Cp2kDiagWorkChain = WorkflowFactory("nanotech_empa.cp2k.diag")
StmCalculation = CalculationFactory('spm.stm')

class Cp2kStmWorkChain(WorkChain):

    @classmethod
    def define(cls, spec):
        super(Cp2kStmWorkChain, cls).define(spec)
        
        spec.input("cp2k_code", valid_type=Code)
        spec.input("structure", valid_type=StructureData)
        spec.input("wfn_file_path", valid_type=Str, required=False)
        spec.input("dft_params", valid_type=Dict)
        spec.input("stm_code", valid_type=Code)
        spec.input("stm_params", valid_type=Dict)
        spec.input("options", valid_type=Dict)
        
        spec.outline(
            cls.setup,            
            cls.run_diag_scf,
            cls.run_stm,
            cls.finalize,
        )
        
        spec.outputs.dynamic = True

        spec.exit_code(
            390,
            "ERROR_TERMINATION",
            message="One or more steps of the work chain failed.",
        )

    def setup(self):
        self.report("Setting up workchain")
        structure = self.inputs.structure
        self.ctx.n_atoms = len(structure.sites)
        emax = float(self.inputs.stm_params.get_dict()['--energy_range'][1])
        added_mos = np.max([100, int(1.2*self.ctx.n_atoms*emax/5.0)])
        self.ctx.dft_params = self.inputs.dft_params.get_dict()
        self.ctx.dft_params["added_mos"] = added_mos
              
    def run_diag_scf(self):
        self.report("Running CP2K diagonalization SCF")        
        builder = Cp2kDiagWorkChain.get_builder()
        builder.cp2k_code = self.inputs.cp2k_code
        builder.structure = self.inputs.structure
        builder.dft_params = Dict(dict=self.ctx.dft_params)
        builder.options = self.inputs.options

        future = self.submit(builder)
        self.to_context(diag_scf=future)
              
    def run_stm(self):
        self.report("STM calculation")
        if not common_utils.check_if_calc_ok(self, self.ctx.diag_scf):
            return self.exit_codes.ERROR_TERMINATION  # pylint: disable=no-member             
        inputs = {}
        inputs['metadata'] = {}
        inputs['metadata']['label'] = "stm"
        inputs['code'] = self.inputs.stm_code
        inputs['parameters'] = self.inputs.stm_params
        inputs['parent_calc_folder'] = self.ctx.diag_scf.outputs.remote_folder
        
        n_machines = 6
        if self.ctx.n_atoms > 1000:
            n_machines = 12
        if self.ctx.n_atoms > 2000:
            n_machines = 18
        if self.ctx.n_atoms > 3000:
            n_machines = 24
        if self.ctx.n_atoms > 4000:
            n_machines = 30
        
        inputs['metadata']['options'] = {
            "resources": {"num_machines": n_machines},
            "max_wallclock_seconds": 36000,
        } 
        if self.inputs.dft_params["protocol"] == "debug":
            inputs['metadata']['options']['max_wallclock_seconds'] = 600
        # Need to make an explicit instance for the node to be stored to aiida
        settings = Dict(dict={'additional_retrieve_list': ['stm.npz']})
        inputs['settings'] = settings
        
        
        future = self.submit(StmCalculation, **inputs)
        return ToContext(stm=future)
    
    def finalize(self):
        if "stm.npz" not in [obj.name for obj in self.ctx.stm.outputs.retrieved.list_objects()]:
            self.report("STM calculation did not finish correctly")
            return self.exit_codes.ERROR_TERMINATION  
        # Add the workchain pk to the input structure extras
        extras_label = "Cp2kStmWorkChain_uuids"
        if extras_label not in self.inputs.structure.extras:
            extras_list = []
        else:
            extras_list = self.inputs.structure.extras[extras_label]
        extras_list.append(self.node.uuid)
        self.inputs.structure.set_extra(extras_label, extras_list)      
        self.report("Work chain is finished")
    
    
# ==========================================================================    
    @classmethod
    def get_options(cls, n_atoms):

        num_machines = 12
        if n_atoms > 500:
            num_machines = 27
        if n_atoms > 1200:
            num_machines = 48
        if n_atoms > 2400:
            num_machines = 60
        if n_atoms > 3600:
            num_machines = 75
        walltime = 86400

        # resources
        options = {
            "resources": {"num_machines": num_machines},
            "max_wallclock_seconds": walltime,
            "append_text": "cp $CP2K_DATA_DIR/BASIS_MOLOPT .",
        }
    

        return options

    # ==========================================================================

