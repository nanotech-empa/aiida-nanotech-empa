from aiida.orm import StructureData
from aiida.orm import Dict
from aiida.orm.nodes.data.array import ArrayData
from aiida.orm import Int, Float, Str, Bool
from aiida.orm import SinglefileData
from aiida.orm import RemoteData
from aiida.orm import Code

from aiida.engine import WorkChain, ToContext, while_
from aiida.engine import submit

from aiida_cp2k.calculations import Cp2kCalculation

from apps.scanning_probe import common

from aiida.plugins import CalculationFactory
AfmCalculation = CalculationFactory('spm.afm')

import os
import tempfile
import shutil
import numpy as np

class AfmWorkChain(WorkChain):

    @classmethod
    def define(cls, spec):
        super(AfmWorkChain, cls).define(spec)
        
        spec.input("cp2k_code", valid_type=Code)
        spec.input("structure", valid_type=StructureData)
        spec.input("wfn_file_path", valid_type=Str, required=False)
        spec.input("dft_params", valid_type=Dict)
        spec.input("options", valid_type=Dict)
        
        spec.input("afm_pp_code", valid_type=Code)
        spec.input("afm_pp_params", valid_type=Dict)
        
        spec.input("afm_2pp_code", valid_type=Code)
        spec.input("afm_2pp_params", valid_type=Dict)
        
        spec.outline(
            cls.setup
            cls.run_diag_scf,
            cls.run_afms,
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
        self.ctx.dft_params = self.inputs.dft_params.get_dict()
        if "smear_t" in self.ctx.dft_params:
            added_mos = np.max([100, int(1.2*n_atoms*2/5.0)])
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

    def run_afms(self):
        self.report("Running PP")
        if not common_utils.check_if_calc_ok(self, self.ctx.diag_scf):
            return self.exit_codes.ERROR_TERMINATION  # pylint: disable=no-member        
        afm_pp_inputs = {}
        afm_pp_inputs['metadata'] = {}
        afm_pp_inputs['metadata']['label'] = "afm_pp"
        afm_pp_inputs['code'] = self.inputs.afm_pp_code
        afm_pp_inputs['parameters'] = self.inputs.afm_pp_params
        afm_pp_inputs['parent_calc_folder'] = self.ctx.diag_scf.outputs.remote_folder
        afm_pp_inputs['atomtypes'] = SinglefileData(file="/home/aiida/apps/scanning_probe/afm/atomtypes_pp.ini")
        afm_pp_inputs['metadata']['options'] = {
            "resources": {"num_machines": 1},
            "max_wallclock_seconds": 7200,
        }
        self.report("Afm pp inputs: " + str(afm_pp_inputs))
        afm_pp_future = self.submit(AfmCalculation, **afm_pp_inputs)
        self.to_context(afm_pp=afm_pp_future)
        
        self.report("Running 2PP")
        
        afm_2pp_inputs = {}
        afm_2pp_inputs['metadata'] = {}
        afm_2pp_inputs['metadata']['label'] = "afm_2pp"
        afm_2pp_inputs['code'] = self.inputs.afm_2pp_code
        afm_2pp_inputs['parameters'] = self.inputs.afm_2pp_params
        afm_2pp_inputs['parent_calc_folder'] = self.ctx.diag_scf.outputs.remote_folder
        afm_2pp_inputs['atomtypes'] = SinglefileData(file="/home/aiida/apps/scanning_probe/afm/atomtypes_2pp.ini")
        afm_2pp_inputs['metadata']['options'] = {
            "resources": {"num_machines": 1},
            "max_wallclock_seconds": 7200,
        }
        self.report("Afm 2pp inputs: " + str(afm_2pp_inputs))
        afm_2pp_future = self.submit(AfmCalculation, **afm_2pp_inputs)
        self.to_context(afm_2pp=afm_2pp_future)
   
    def finalize(self):
        retrieved_list = [obj.name for obj in self.ctx.stm.outputs.retrieved.list_objects()]
        if "df.npy" not in retrieved_list or "df_vec.npy" not in retrieved_list::
            self.report("AFM calculation did not finish correctly")
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
        self.report("Work chain is finished")
    
    
