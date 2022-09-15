import os
import pathlib
import copy
import yaml

from aiida_nanotech_empa.workflows.cp2k.cp2k_utils import get_kinds_section, determine_kinds
from aiida_nanotech_empa.workflows.cp2k.cp2k_utils import dict_merge, get_nodes, get_cutoff, get_colvars_section, get_constraints_section
from aiida_nanotech_empa.utils import common_utils
from aiida_nanotech_empa.workflows.cp2k.cp2k_utils import compute_colvars
from aiida.plugins import DataFactory, WorkflowFactory
from aiida.engine import WorkChain, ToContext, ExitCode, while_, if_, append_, calcfunction
from aiida.orm import SinglefileData,Int, Str, Code, Dict, List, Bool, load_node, CalcJobNode
import math
import numpy as np

StructureData = DataFactory("structure")
Cp2kBaseWorkChain = WorkflowFactory("cp2k.base")
#Cp2kOptWorkChain = WorkflowFactory('nanotech_empa.cp2k.opt')

@calcfunction
def output_dict(enes,cvs,structures):

    return Dict(dict={
        'energies': enes,
        'cvs': listcvs,
        'structures':structures
    })

class Cp2kReplicaWorkChain(WorkChain):
    """Workflow to run Replica Chain calculations with CP2K."""
    @classmethod
    def define(cls, spec):
        """Define the workflow."""
        super().define(spec)

        # Define the inputs of the workflow
        spec.input(
            "charge",  # +1 means one electron removed
            valid_type=Int,
            default=lambda:         Int(0),
            required=False)
        spec.input("multiplicity",
                   valid_type=Int,
                   default=lambda: Int(0),
                   required=False)
        spec.input("magnetization_per_site",
                   valid_type=List,
                   default=lambda: List(list=[]),
                   required=False)
        spec.input("vdw",
                   valid_type=Bool,
                   default=lambda: Bool(False),
                   required=False)
        spec.input("protocol",
                   valid_type=Str,
                   default=lambda: Str('standard'),
                   required=False,
                   help="Settings to run simulations with.")
        spec.input("walltime_seconds",
                   valid_type=Int,
                   default=lambda: Int(7200),
                   required=False)     
        spec.input("max_nodes",
                   valid_type=Int,
                   default=lambda: Int(48),
                   required=False)                                 
        spec.input("structure", valid_type=StructureData)
        spec.input("code", valid_type=Code)
        spec.input("constraints", valid_type=Str)
        spec.input("colvars", valid_type=Str)
        spec.input("colvars_targets", valid_type=List)
        spec.input("colvars_increments", valid_type=List)
        spec.input("continuation_of",
                   valid_type=Int,
                   default=lambda: Int(0),
                   required=False)

        spec.outline(
            cls.start, 
            if_(cls.should_run_scf)(cls.first_scf, cls.update_colvars_values,
            cls.update_colvars_increments,
            cls.to_outputs),
            while_(cls.should_run_simulations)(
                cls.run_constrained_geo_opts,
                cls.update_latest_structure,
                cls.update_colvars_values,
                cls.update_colvars_increments,
                cls.to_outputs,
            ),
            cls.finalize)

        spec.outputs.dynamic = True
        spec.output_namespace("structures",valid_type=StructureData)
        spec.output_namespace("details",valid_type=Dict) 
        spec.exit_code(390, "ERROR_TERMINATION", message="One geo opt failed")

    def start(self):
        """Initialize the workchain process."""
        self.report(f"Initialization")
        # restart from previous workchain
        if self.inputs.continuation_of.value !=0 :
            self.report(f"Retrieving previous steps from {self.inputs.continuation_of.value} and continuing")
            previous_replica=load_node(self.inputs.continuation_of.value)
            previous_structures=[struc for struc in previous_replica.outputs.structures]
            
            previous_structures.sort()
            self.ctx.latest_structure = previous_replica.outputs.structures[previous_structures[-1]]
            for struc in previous_structures:
                self.out(f"structures.{struc}",previous_replica.outputs.structures[struc])
                self.out(f"details.{struc}",previous_replica.outputs.details[struc])
            self.ctx.CVs_to_increment = previous_replica.outputs.details[struc]['cvs_target']
            self.ctx.colvars_values = previous_replica.outputs.details[struc]['cvs_actual']
            self.ctx.should_run_scf = False           
            self.ctx.propagation_step = len(previous_structures) - 1 # continue form this step
            self.update_colvars_increments()
            self.ctx.should_run_simulations = True
            self.ctx.restart_folder = list(self.ctx.latest_structure.get_incoming(node_class = CalcJobNode))[-1][0].outputs.remote_folder
            self.report(f"data from workchain: {previous_replica.pk}")
            self.report(f"actual CVs: {self.ctx.colvars_values}")
            self.report(f"CVs to increment: {self.ctx.CVs_to_increment}")
            self.report(f"starting from geometry: {self.ctx.latest_structure.pk}")
            self.report(f"retrieved the following steps: {previous_structures} ")
        else:
            self.ctx.latest_structure = self.inputs.structure
            self.ctx.should_run_scf = True
            self.ctx.should_run_simulations = True
            self.ctx.propagation_step = 0
        return ExitCode(0)

    def should_run_scf(self):
        """Function that returnns whether to run or not the first scf step"""
        return self.ctx.should_run_scf        

    def should_run_simulations(self):
        """Function that returnns whether targets have been reached or not"""
        return self.ctx.should_run_simulations

    def to_outputs(self):
        """Function to update step by step the workcain output """
        if self.ctx.propagation_step == 0 :
            self.out('details.initial_scf',Dict(dict={'energy_scf':self.ctx.initial_scf.outputs.output_parameters['energy_scf'],'cvs_target':self.ctx.colvars_values,'cvs_actual':self.ctx.colvars_values}).store())
            self.out('structures.initial_scf',self.ctx.latest_structure)
            self.report(f"Updated output for the initial_scf step")
        else:
            self.out(f"details.step_{self.ctx.propagation_step - 1 :04}",Dict(dict={'energy_scf':self.ctx.lowest_energy,'cvs_target': self.ctx.CVs_cases[self.ctx.lowest_energy_calc],'cvs_actual':self.ctx.colvars_values}).store())
            self.out(f"structures.step_{self.ctx.propagation_step - 1 :04}",self.ctx.latest_structure)
            self.report(f"Updated output for step {self.ctx.propagation_step - 1 :04}")
        return ExitCode(0)

    def first_scf(self):
        """Run scf on the initial geometry."""
        with open(pathlib.Path(__file__).parent /
            './protocols/scf_protocol.yml',
            encoding='utf-8') as handle:
            protocols = yaml.safe_load(handle)
            input_dict = copy.deepcopy(protocols[self.inputs.protocol.value])



        structure = self.ctx.latest_structure
        #cutoff
        self.ctx.cutoff = get_cutoff(structure=structure)

        #get initial magnetization
        magnetization_per_site = copy.deepcopy(
            self.inputs.magnetization_per_site)
        structure_with_tags, kinds_dict = determine_kinds(
            structure, magnetization_per_site)

        ase_atoms = structure_with_tags.get_ase()

        builder = Cp2kBaseWorkChain.get_builder()
        builder.cp2k.structure = StructureData(ase=ase_atoms)
        builder.cp2k.code = self.inputs.code

        builder.cp2k.file = {
            'basis':
            SinglefileData(
                file=os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  ".", "data", "BASIS_MOLOPT")),
            'pseudo':
            SinglefileData(
                file=os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  ".", "data", "POTENTIAL")),
        }

        #charge
        if self.inputs.charge != 0:
            input_dict['FORCE_EVAL']['DFT']['CHARGE'] = self.inputs.charge

        # vdw
        if not self.inputs.vdw.value:
            input_dict['FORCE_EVAL']['DFT']['XC'].pop('VDW_POTENTIAL')

        #UKS
        if self.inputs.multiplicity.value > 0:
            input_dict['FORCE_EVAL']['DFT']['UKS'] = '.TRUE.'
            input_dict['FORCE_EVAL']['DFT'][
                'MULTIPLICITY'] = self.inputs.multiplicity.value

        #cutoff
        input_dict['FORCE_EVAL']['DFT']['MGRID']['CUTOFF'] = self.ctx.cutoff

        # KINDS section
        self.ctx.kinds_section = get_kinds_section(kinds_dict, protocol='gpw')
        dict_merge(input_dict, self.ctx.kinds_section)

        #computational resources
        nodes, tasks_per_node, threads = get_nodes(
            atoms=ase_atoms,
            calctype='slab',
            computer=self.inputs.code.computer,
            max_nodes=self.inputs.max_nodes.value,
            uks=self.inputs.multiplicity.value > 0)

        builder.cp2k.metadata.options.resources = {
            'num_machines': nodes,
            'num_mpiprocs_per_machine': tasks_per_node,
            'num_cores_per_mpiproc': threads
        }

        #walltime
        input_dict['GLOBAL']['WALLTIME'] = max(
            self.inputs.walltime_seconds.value - 600, 600)
        builder.cp2k.metadata.options.max_wallclock_seconds = self.inputs.walltime_seconds.value

        #parser
        builder.cp2k.metadata.options.parser_name = "cp2k_advanced_parser"

        #handlers
        builder.handler_overrides = Dict(
            dict={'resubmit_unconverged_geometry': True})

        #cp2k input dictionary
        builder.cp2k.parameters = Dict(dict=input_dict)

        submitted_calculation = self.submit(builder)
        self.report(
            f"Submitted scf of the initial geometry: {submitted_calculation.pk}")
        return ToContext(initial_scf=submitted_calculation)

    def update_colvars_values(self):
        """Compute actual value of CVs."""

        ase_structure = self.ctx.latest_structure.get_ase()
        colvars = self.inputs.colvars.value
        self.ctx.colvars_values = [cv[1] for cv in compute_colvars(colvars, ase_structure)]
        self.report(f"actual CVs values: {self.ctx.colvars_values}")
        if self.ctx.propagation_step == 0 :
            self.ctx.CVs_to_increment = self.ctx.colvars_values
        else:
            self.ctx.CVs_to_increment = self.ctx.CVs_cases[self.ctx.lowest_energy_calc]
            #self.report(f"will add increments to this set of CVs: {self.ctx.CVs_to_increment}")

    def update_colvars_increments(self):
        """Computes teh increments for the CVs according to deviation from target. 
        If the target value is reached wihtin the increment, set increment to 0.
        Deviation from target is computed wrt actual value of CVs while new CVs 
        are computed as previous target plus increment to avoid slow diverging deviations 
        from targets"""
        self.ctx.colvars_increments = []
        for index, colvar in enumerate(self.ctx.colvars_values):
            if math.fabs(self.inputs.colvars_targets[index] -
                        colvar) > self.inputs.colvars_increments[index] and math.fabs(self.inputs.colvars_increments[index]) > 0.0001:
                self.ctx.colvars_increments.append(
                    math.fabs(self.inputs.colvars_increments[index]) *
                    np.sign(self.inputs.colvars_targets[index] - colvar))
            else:
                self.ctx.colvars_increments.append(0)
        if all(i == 0 for i in self.ctx.colvars_increments):
            self.ctx.should_run_simulations = False

    def run_constrained_geo_opts(self):
        """Run a constrained geometry optimization for each non 0 increment of colvars."""
        #pylint: disable=unused-variable
        self.ctx.CVs_cases=[]
        for index, value in enumerate(self.ctx.CVs_to_increment): #(self.ctx.colvars_values):
            if self.ctx.colvars_increments[index] != 0:
                with open(pathlib.Path(__file__).parent /
                        './protocols/geo_opt_protocol.yml',
                        encoding='utf-8') as handle:
                    protocols = yaml.safe_load(handle)
                    input_dict = copy.deepcopy(protocols[self.inputs.protocol.value])

                structure = self.ctx.latest_structure
                #cutoff
                self.ctx.cutoff = get_cutoff(structure=structure)

                #get initial magnetization
                magnetization_per_site = copy.deepcopy(
                    self.inputs.magnetization_per_site)
                structure_with_tags, kinds_dict = determine_kinds(
                    structure, magnetization_per_site)

                ase_atoms = structure_with_tags.get_ase()        
                builder = Cp2kBaseWorkChain.get_builder()
                builder.cp2k.structure = StructureData(ase=ase_atoms)
                builder.cp2k.code = self.inputs.code

                builder.cp2k.file = {
                    'basis':
                    SinglefileData(
                        file=os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                        ".", "data", "BASIS_MOLOPT")),
                    'pseudo':
                    SinglefileData(
                        file=os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                        ".", "data", "POTENTIAL")),
                }

                #charge
                if self.inputs.charge != 0:
                    input_dict['FORCE_EVAL']['DFT']['CHARGE'] = self.inputs.charge

                # vdw
                if not self.inputs.vdw.value:
                    input_dict['FORCE_EVAL']['DFT']['XC'].pop('VDW_POTENTIAL')

                #UKS
                if self.inputs.multiplicity.value > 0:
                    input_dict['FORCE_EVAL']['DFT']['UKS'] = '.TRUE.'
                    input_dict['FORCE_EVAL']['DFT'][
                        'MULTIPLICITY'] = self.inputs.multiplicity.value

                #constraints
                input_dict['MOTION']['CONSTRAINT'] = get_constraints_section(self.inputs.constraints.value)
                submitted_CVs=''
                current_CVs_targets=[]
                for icv, cvval in enumerate(self.ctx.CVs_to_increment):
                    target = cvval
                    units = input_dict['MOTION']['CONSTRAINT']['COLLECTIVE'][icv]['TARGET'].split(' ')[0]
                    if icv == index:                       
                        target += self.ctx.colvars_increments[icv]
                    current_CVs_targets.append(target)
                    input_dict['MOTION']['CONSTRAINT']['COLLECTIVE'][icv]['TARGET'] = units + ' ' + str(target)
                    submitted_CVs += ' ' + str(target)
                self.ctx.CVs_cases.append(current_CVs_targets)
                #colvars
                if self.inputs.colvars.value:
                    input_dict['FORCE_EVAL']['SUBSYS'].update(
                        get_colvars_section(self.inputs.colvars.value))

                #cutoff
                input_dict['FORCE_EVAL']['DFT']['MGRID']['CUTOFF'] = self.ctx.cutoff

                # KINDS section
                self.ctx.kinds_section = get_kinds_section(kinds_dict, protocol='gpw')
                dict_merge(input_dict, self.ctx.kinds_section)

                #computational resources
                nodes, tasks_per_node, threads = get_nodes(
                    atoms=ase_atoms,
                    calctype='slab',
                    computer=self.inputs.code.computer,
                    max_nodes=self.inputs.max_nodes.value,
                    uks=self.inputs.multiplicity.value > 0)

                builder.cp2k.metadata.options.resources = {
                    'num_machines': nodes,
                    'num_mpiprocs_per_machine': tasks_per_node,
                    'num_cores_per_mpiproc': threads
                }

                #walltime
                input_dict['GLOBAL']['WALLTIME'] = max(
                    self.inputs.walltime_seconds.value - 600, 600)
                builder.cp2k.metadata.options.max_wallclock_seconds = self.inputs.walltime_seconds.value

                #parser
                builder.cp2k.metadata.options.parser_name = "cp2k_advanced_parser"

                #handlers
                builder.handler_overrides = Dict(
                    dict={'resubmit_unconverged_geometry': True})

                #wfn restart folder
                if self.ctx.propagation_step == 0:
                    builder.cp2k.parent_calc_folder = self.ctx.initial_scf.outputs.remote_folder 
                else:
                    builder.cp2k.parent_calc_folder = self.ctx.restart_folder

                #cp2k input dictionary
                builder.cp2k.parameters = Dict(dict=input_dict)


                submitted_calculation = self.submit(builder)
                self.report(
                    f"Submitted geo opt: {submitted_calculation.pk}, with {submitted_CVs}"
                )
                self.to_context(
                    **{f"run_{self.ctx.propagation_step :04}": append_(submitted_calculation)})

    def update_latest_structure(self):
        """Update the latest structure as the one with minimum energy from the constrained 
        geometry optimizations."""
        results = []
        for index, calculation in enumerate(
                getattr(self.ctx, f"run_{self.ctx.propagation_step :04}")):
            #check if the calculation is finished
            if not common_utils.check_if_calc_ok(self,calculation):
                return self.exit_codes.ERROR_TERMINATION  # pylint: disable=no-member
            results.append((calculation.outputs.output_parameters['energy_scf'], index))
        self.report(f"energies {results}")
        results.sort(key=lambda x: x[0])  #pylint: disable=expression-not-assigned
        self.ctx.lowest_energy_calc = results[0][1]
        self.ctx.latest_structure = getattr(
            self.ctx, f"run_{self.ctx.propagation_step :04}"
        )[self.ctx.lowest_energy_calc].outputs.output_structure
        self.ctx.lowest_energy = results[0][0]
        self.report(
            f"The lowest energy at step {self.ctx.propagation_step :04} is {self.ctx.lowest_energy}"
        )
        self.report(f"geometry: {self.ctx.latest_structure.pk}")
        self.report(f"target CVs {self.ctx.CVs_cases[self.ctx.lowest_energy_calc]}")
        #define restart folder
        self.ctx.restart_folder = getattr(
            self.ctx, f"run_{self.ctx.propagation_step :04}"
        )[self.ctx.lowest_energy_calc].outputs.remote_folder

        #increment step index
        self.ctx.propagation_step += 1
        return ExitCode(0)

    def finalize(self):
        self.report("Finalizing...")
        #self.out('output_parameters',Dict(dict={'energies':self.ctx.outenes,'cvs':self.ctx.outcvs,'structures':self.ctx.outstructures}).store())
        return ExitCode(0)        