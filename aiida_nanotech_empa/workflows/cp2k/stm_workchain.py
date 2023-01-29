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

Cp2kBaseWorkChain = WorkflowFactory("cp2k.base")
StmCalculation = CalculationFactory('spm.stm')

class Cp2kStmWorkChain(WorkChain):

    @classmethod
    def define(cls, spec):
        super(Cp2kStmWorkChain, cls).define(spec)
        
        spec.input("cp2k_code", valid_type=Code)
        spec.input("structure", valid_type=StructureData)
        spec.input("wfn_file_path", valid_type=Str, default=lambda: Str(""))
        spec.input("dft_params", valid_type=Dict)
        spec.input("stm_code", valid_type=Code)
        spec.input("stm_params", valid_type=Dict)
        
        spec.outline(
            cls.setup,
            cls.run_ot_scf,            
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
        self.ctx.files = {
            "basis": SinglefileData(
                file=os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    ".",
                    "data",
                    "BASIS_MOLOPT",
                )
            ),
            "pseudo": SinglefileData(
                file=os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    ".",
                    "data",
                    "POTENTIAL",
                )
            ),
        }

        structure = self.inputs.structure
        self.ctx.n_atoms = len(structure.sites)
        self.report("Number of atoms: {}".format(self.ctx.n_atoms))
        # set up mol UKS parameters

        self.ctx.dft_params = copy.deepcopy(self.inputs.dft_params.get_dict())

        if "uks" not in self.ctx.dft_params:
            self.ctx.dft_params["uks"] = False
            self.ctx.dft_params["spin_up_guess"] = []
            self.ctx.dft_params["spin_dw_guess"] = []

        
        # cutoff: use the same for all calculations
        self.ctx.cutoff = get_cutoff(structure=structure)

        # get initial magnetization
        spin_up_guess = self.ctx.dft_params["spin_up_guess"]
        spin_dw_guess = self.ctx.dft_params["spin_dw_guess"]
        magnetization_per_site = [
            1
            if i in spin_up_guess
            else -1
            if i in spin_dw_guess
            else 0
            for i in range(self.ctx.n_atoms)
        ]
        structure_with_tags, kinds_dict = determine_kinds(
            structure, magnetization_per_site
        )

        ase_atoms = structure_with_tags.get_ase()

        self.ctx.structure_with_tags = ase_atoms  
        self.ctx.kinds_section = get_kinds_section(kinds_dict, protocol="gpw")          

    def run_ot_scf(self):
        self.report("Running CP2K OT SCF")

        # load input template
        with open(
            pathlib.Path(__file__).parent / "./protocols/scf_ot_protocol.yml",
            encoding="utf-8",
        ) as handle:
            protocols = yaml.safe_load(handle)
            input_dict = copy.deepcopy(protocols[self.ctx.dft_params['protocol']])

        builder = Cp2kBaseWorkChain.get_builder()
        builder.cp2k.code = self.inputs.cp2k_code
        builder.cp2k.structure = StructureData(ase=self.ctx.structure_with_tags)

        builder.cp2k.file = self.ctx.files
        if self.inputs.wfn_file_path != "":
            builder.cp2k.parent_calc_folder = self.inputs.wfn_file_path.value

        input_dict["FORCE_EVAL"]["DFT"]["XC"].pop("VDW_POTENTIAL")

        # UKS
        if self.ctx.dft_params["uks"]:
            input_dict["FORCE_EVAL"]["DFT"]["UKS"] = ".TRUE."
            input_dict["FORCE_EVAL"]["DFT"]["MULTIPLICITY"] = self.ctx.dft_params[
                "multiplicity"
            ]

        # cutoff
        input_dict["FORCE_EVAL"]["DFT"]["MGRID"]["CUTOFF"] = self.ctx.cutoff

        # KINDS section
        dict_merge(input_dict, self.ctx.kinds_section)

        # Setup walltime.
        input_dict["GLOBAL"]["WALLTIME"] = 86000

        self.ctx.options = self.get_options(self.ctx.n_atoms)
        if self.ctx.dft_params['protocol'] == "debug":
            self.ctx.options = {
            "resources": {"num_machines": 1,
            "num_mpiprocs_per_machine": 8,
            "num_cores_per_mpiproc": 1,},
            "max_wallclock_seconds": 600,
            "append_text": "cp $CP2K_DATA_DIR/BASIS_MOLOPT .",
        }
        builder.cp2k.metadata.options = self.ctx.options

        # parser
        builder.cp2k.metadata.options.parser_name = "cp2k_advanced_parser"

        # cp2k input dictionary
        builder.cp2k.parameters = Dict(dict=input_dict)
        self.ctx.input_dict = copy.deepcopy(input_dict)

        future = self.submit(builder)
        self.to_context(ot_scf=future)

    
    def run_diag_scf(self):
        emax = float(self.inputs.stm_params.get_dict()['--energy_range'][1])
        self.report("Running CP2K diagonalization SCF")
        if not common_utils.check_if_calc_ok(self, self.ctx.ot_scf):
            return self.exit_codes.ERROR_TERMINATION  # pylint: disable=no-member

        added_mos = np.max([100, int(1.2*self.ctx.n_atoms*emax/5.0)])
        self.report(f"Adding {added_mos} MOs to the basis set")

        # load input template
        with open(
            pathlib.Path(__file__).parent / "./protocols/scf_diag_protocol.yml",
            encoding="utf-8",
        ) as handle:
            protocols = yaml.safe_load(handle)
            scf_dict = copy.deepcopy(protocols[self.ctx.dft_params['protocol']])

        input_dict = copy.deepcopy(self.ctx.input_dict)
        if self.ctx.dft_params["elpa_switch"]:
            input_dict["GLOBAL"]["PREFERRED_DIAG_LIBRARY"] = "ELPA"
            input_dict["GLOBAL"]["ELPA_KERNEL"] = "AUTO"
            input_dict["GLOBAL"]["DBCSR"] = {"USE_MPI_ALLOCATOR": ".FALSE."}
        input_dict["FORCE_EVAL"]["DFT"].pop("SCF")
        input_dict["FORCE_EVAL"]["DFT"]["SCF"] = scf_dict
        input_dict["FORCE_EVAL"]["DFT"]["SCF"]["ADDED_MOS"] = added_mos

        smearing = "smear_t" in self.ctx.dft_params
        if smearing:
            input_dict["FORCE_EVAL"]["DFT"]["SCF"]["SMEAR"][
                "ELECTRONIC_TEMPERATURE"
            ] = self.ctx.dft_params["smear_t"]
        else:
            input_dict["FORCE_EVAL"]["DFT"]["SCF"].pop("SMEAR")

        # UKS
        if self.ctx.dft_params["uks"] and smearing and self.ctx.dft_params['sc_diag']:
            input_dict["FORCE_EVAL"]["DFT"]["SCF"]["SMEAR"][
                "FIXED_MAGNETIC_MOMENT"
            ] = (self.ctx.dft_params["multiplicity"] - 1)
        # no self consistent diag
        if not self.ctx.dft_params['sc_diag']:
            input_dict["FORCE_EVAL"]["DFT"]["SCF"].pop("SMEAR")
            input_dict["FORCE_EVAL"]["DFT"]["SCF"]["EPS_SCF"] = "1.0E-1"
            input_dict["FORCE_EVAL"]["DFT"]["SCF"]["OUTER_SCF"]["EPS_SCF"] = "1.0E-1"

        if not smearing and "SMEAR" in input_dict["FORCE_EVAL"]["DFT"]["SCF"]:
            input_dict["FORCE_EVAL"]["DFT"]["SCF"].pop("SMEAR")

        builder = Cp2kBaseWorkChain.get_builder()
        builder.cp2k.code = self.inputs.cp2k_code
        builder.cp2k.structure = StructureData(ase=self.ctx.structure_with_tags)

        builder.cp2k.file = self.ctx.files
        builder.cp2k.settings = Dict(dict={"additional_retrieve_list": [
            'aiida.inp', 'BASIS_MOLOPT', 'aiida.coords.xyz', 'aiida-RESTART.wfn'
        ]})
        builder.cp2k.parent_calc_folder = self.ctx.ot_scf.outputs.remote_folder

        builder.cp2k.metadata.options = self.ctx.options

        # parser
        builder.cp2k.metadata.options.parser_name = "cp2k_advanced_parser"

        # cp2k input dictionary
        builder.cp2k.parameters = Dict(dict=input_dict)

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
        if self.inputs.dft_params["protocol"] == "debug"
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

