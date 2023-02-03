import copy
import os
import pathlib

import numpy as np
import yaml

from aiida.engine import ToContext, WorkChain,  while_
from aiida.orm import Bool, Code, Dict, List, SinglefileData, Str, StructureData
from aiida.orm.nodes.data.array import ArrayData
from aiida.orm import SinglefileData
from aiida.orm import RemoteData

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

class Cp2kOrbitalsWorkChain(WorkChain):

    @classmethod
    def define(cls, spec):
        super(Cp2kOrbitalsWorkChain, cls).define(spec)
        
        spec.input("cp2k_code", valid_type=Code)
        spec.input("structure", valid_type=StructureData)
        spec.input("wfn_file_path", valid_type=Str, required=False)                
        spec.input("dft_params", valid_type=Dict)        
        spec.input("stm_code", valid_type=Code)
        spec.input("stm_params", valid_type=Dict)
        
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
        n_lumo = int(self.inputs.stm_params.get_dict()['--n_lumo'])
        added_mos = np.max([n_lumo,20])
        self.ctx.dft_params = self.inputs.dft_params.get_dict()
        self.ctx.dft_params["added_mos"] = added_mos
        self.ctx.options = self.get_options(len(structure.sites))
        
    def run_diag_scf(self):
        self.report("Running CP2K diagonalization SCF")        
        builder = Cp2kDiagWorkChain.get_builder()
        builder.cp2k_code = self.inputs.cp2k_code
        builder.structure = self.inputs.structure
        builder.dft_params = Dict(dict=self.ctx.dft_params)
        builder.settings = Dict(dict={'additional_retrieve_list': [
            'aiida.inp', 'BASIS_MOLOPT', 'aiida.coords.xyz', 'aiida-RESTART.wfn'
        ]})
        builder.options = Dict(dict=self.ctx.options)

        future = self.submit(builder)
        self.to_context(diag_scf=future)
              
    def run_stm(self):
        self.report("STM calculation")
        if not common_utils.check_if_calc_ok(self, self.ctx.diag_scf):
            return self.exit_codes.ERROR_TERMINATION  # pylint: disable=no-member

        inputs = {}
        inputs['metadata'] = {}
        inputs['metadata']['label'] = "orb"
        inputs['code'] = self.inputs.stm_code
        inputs['parameters'] = self.inputs.stm_params
        inputs['parent_calc_folder'] = self.ctx.diag_scf.outputs.remote_folder
        inputs['metadata']['options'] = {
            "resources": {"num_machines": 1},
            "max_wallclock_seconds": 3600,
        } 
        
        # Need to make an explicit instance for the node to be stored to aiida
        settings = Dict(dict={'additional_retrieve_list': ['orb.npz']})
        inputs['settings'] = settings
        
        future = self.submit(StmCalculation, **inputs)
        return ToContext(stm=future)
    
    def finalize(self):
        if "orb.npz" not in [obj.name for obj in self.ctx.stm.outputs.retrieved.list_objects()]:
            self.report("Orbital calculation did not finish correctly")
            return self.exit_codes.ERROR_TERMINATION
        # Add the workchain pk to the input structure extras
        extras_label = "Cp2kOrbitalsWorkChain_uuids"
        if extras_label not in self.inputs.structure.extras:
            extras_list = []
        else:
            extras_list = self.inputs.structure.extras[extras_label]
        extras_list.append(self.node.uuid)
        self.inputs.structure.set_extra(extras_label, extras_list)
        self.report("Work chain is finished")
    
    
     # ==========================================================================

# ==========================================================================    
    @classmethod
    def get_options(cls, n_atoms):

        num_machines = 3
        if n_atoms > 50:
            num_machines = 6
        if n_atoms > 100:
            num_machines = 12
        if n_atoms > 300:
            num_machines = 27
        if n_atoms > 650:
            num_machines = 48
        walltime = 72000

        # resources
        options = {
            "resources": {"num_machines": num_machines},
            "max_wallclock_seconds": walltime,
            "append_text": "cp $CP2K_DATA_DIR/BASIS_MOLOPT .",
        }
    

        return options

    # ==========================================================================