"""Workflow to run Replica Chain calculations with CP2K."""
import os
import pathlib
import copy
import yaml

from aiida_nanotech_empa.workflows.cp2k.cp2k_utils import get_kinds_section, determine_kinds
from aiida_nanotech_empa.workflows.cp2k.cp2k_utils import dict_merge, get_nodes, get_cutoff, get_colvars_section, get_constraints_section
from aiida_nanotech_empa.utils import common_utils
from aiida_nanotech_empa.workflows.cp2k.cp2k_utils import compute_colvars
from aiida.plugins import DataFactory
from aiida.engine import WorkChain, ToContext, ExitCode, while_, append_
from aiida.orm import SinglefileData,Int, Str, Code, Dict, List, Bool
import math
import numpy as np

StructureData = DataFactory("structure")
Cp2kBaseWorkChain = DataFactory("cp2k.base")
#Cp2kOptWorkChain = WorkflowFactory('nanotech_empa.cp2k.opt')


class Cp2kReplicaWorkChain(WorkChain):
    """Workflow to run Replica Chain calculations with CP2K."""
    @classmethod
    def define(cls, spec):
        """Define the workflow."""

        # Define the inputs of the workflow
        spec.input(
            "charge",  # +1 means one electron removed
            valid_type=Int,
            default=lambda: Int(0),
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
            cls.start, cls.first_scf, cls.update_colvars_values,
            cls.update_colvars_increments,
            while_(cls.should_run_simulations)(
                cls.run_constrained_geo_opts,
                cls.update_latest_structure,
                cls.update_colvars_values,
                cls.update_colvars_increments,
            ))

        spec.exit_code(390, "ERROR_TERMINATION", message="One geo opt failed")

    def start(self):
        """Initialise the workchain process."""
        self.ctx.latest_structure = self.inputs.structure
        self.ctx.should_run_simulations = True
        self.propagation_step = 0

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
        builder.structure = StructureData(ase=ase_atoms)
        builder.code = self.inputs.code

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
            f"Submitted scf of the initial geometry: {submitted_calculation}")
        return ToContext(initial_scf=submitted_calculation)

    def update_colvars_values(self):
        ase_structure = self.ctx.latest_structure.get_ase()
        colvars = self.inputs.colvars.value
        self.ctx.colvars_values = compute_colvars(colvars, ase_structure)

    def update_colvars_increments(self):
        """Increment/decrement the colvars according to deviation from target. 
        If the target value is reached, set increment to 0."""
        self.ctx.colvar_increments = []
        for index, colvar in enumerate(self.ctx.colvars_values):
            if math.fabs(self.inputs.colvars_targets[index] -
                        colvar) > self.inputs.colvars_increments[index]:
                self.ctx.colvars_increments.append(
                    math.fabs(self.inputs.colvars_increments[index]) *
                    np.sign(self.inputs.colvar_targets[index] - colvar))
            else:
                self.ctx.colvars_increments.append(0)
        if all(i == 0 for i in self.ctx.colvar_increments):
            self.ctx.should_run_simulations = False

    def run_constrained_geo_opts(self):
        """Run a constrained geometry optimization for each non 0 increment of colvars."""
        #pylint: disable=unused-variable
        for index, value in enumerate(self.ctx.colvars_values):
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
                builder.structure = StructureData(ase=ase_atoms)
                builder.code = self.inputs.code

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

                # vdw
                if not self.inputs.vdw.value:
                    input_dict['FORCE_EVAL']['DFT']['XC'].pop('VDW_POTENTIAL')

                #UKS
                if self.inputs.multiplicity.value > 0:
                    input_dict['FORCE_EVAL']['DFT']['UKS'] = '.TRUE.'
                    input_dict['FORCE_EVAL']['DFT'][
                        'MULTIPLICITY'] = self.inputs.multiplicity.value

                #constraints
                if self.inputs.constraints.value:
                    input_dict['MOTION']['CONSTRAINT'] = get_constraints_section(self.inputs.constraints.value)
                    for icv, cvval in enumerate(self.ctx.colvars_values):
                        target = self.ctx.colvars_values
                        units = input_dict['MOTION']['CONSTRAINT'][icv]['TARGET'].split(' ')[0]
                        if icv == index:                       
                            target += self.ctx.colvars_increments[icv]
                        input_dict['MOTION']['CONSTRAINT'][icv]['TARGET'] = units + ' ' + str(target)

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

                #cp2k input dictionary
                builder.cp2k.parameters = Dict(dict=input_dict)


                submitted_calculation = self.submit(builder)
                self.report(
                    f"Submitted constrained geometry optimization: {submitted_calculation}"
                )
                self.to_context(
                    **{f"run_{self.propagation_step}": append_(submitted_calculation)})

    def update_latest_structure(self):
        """Update the latest structure as teh one with minimum energy from the constrained 
        geometry optimizations."""
        results = []
        for index, calculation in enumerate(
                getattr(self.ctx, f"run_{self.propagation_step}")):
            #check if the calculation is finished
            if not common_utils.check_if_calc_ok(self,calculation):
                return self.exit_codes.ERROR_TERMINATION  # pylint: disable=no-member
            results.append((calculation.outputs.total_energy, index))
        results.sort(key=lambda x: x[0])[0][1]  #pylint: disable=expression-not-assigned
        lowest_energy_calc = results[0][1]
        self.ctx.latest_structure = getattr(
            self.ctx, f"run_{self.propagation_step}"
        )[lowest_energy_calc].outputs.structure
        self.report(
            f"The lowest energy at step {self.ctx.propagation_step} is {results[0][0]}"
        )
        self.ctx.propagation_step += 1
        return ExitCode(0)
