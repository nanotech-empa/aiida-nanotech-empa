import os
import pathlib
import yaml
import copy
import numpy as np

from aiida.engine import WorkChain, ToContext
from aiida.orm import Int, Float, Str, Code, Dict, List
from aiida.orm import SinglefileData, StructureData
from aiida_nanotech_empa.utils.cp2k_utils import get_kinds_section_gw, tags_and_magnetization_gw, dict_merge, get_nodes, get_cutoff
from aiida_cp2k.calculations import Cp2kCalculation

PROTOCOLS = {
    'gw_st': ['gw_st_first_step', 'gw_st_second_step'],
    'gw_st_ic': ['gw_st_first_step', 'gw_st_second_step'],
    'gw': ['gw_first_step', 'gw_second_step'],
    'gw_ic': ['gw_first_step', 'gw_ic_second_step'],
    'gw_hq': ['gw_hq_first_step', 'gw_hq_second_step'],
    'gw_hq_ic': ['gw_hq_first_step', 'gw_ic_hq_second_step']
}


class Cp2kMoleculeGwWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("code", valid_type=Code)
        spec.input("gw_type",
                   valid_type=Str,
                   default=lambda: Str('gw'),
                   required=False)
        spec.input("z_ic_plane",
                   valid_type=Float,
                   default=lambda: Float(8.22),
                   required=False)
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
        spec.input("walltime_seconds",
                   valid_type=Int,
                   default=lambda: Int(300),
                   required=False)
        spec.input("structure", valid_type=StructureData)

        spec.outline(cls.setup, cls.submit_first_step, cls.submit_second_step,
                     cls.finalize)
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
        """Function to submit the first step of the workchain."""
        #load input template
        with open(
                pathlib.Path(__file__).parent /
                '../../files/cp2k/gw_protocols.yml') as handle:
            self.ctx.protocols = yaml.safe_load(handle)
            input_dict = copy.deepcopy(
                self.ctx.protocols[PROTOCOLS[self.inputs.gw_type.value][0]])

        structure = self.inputs.structure
        self.ctx.cutoff = get_cutoff(structure=structure)
        magnetization_per_site = copy.deepcopy(
            self.inputs.magnetization_per_site)
        ghosts = []
        #add ghost atoms in case of gw-ic
        if 'ic' in self.inputs.gw_type.value:
            atoms = self.inputs.structure.get_ase()
            image = atoms.copy()
            image.set_masses([int(999) for a in range(len(image))])
            image.positions[:, 2] = (2 * self.inputs.z_ic_plane.value -
                                     atoms.positions[:, 2])
            ghosts = [0 for a in atoms] + [1 for a in image]
            if (magnetization_per_site):
                magnetization_per_site += [int(0) for i in range(len(image))]
            structure = StructureData(ase=atoms + image)

        #get initial magnetization
        structure_with_tags = tags_and_magnetization_gw(
            structure, magnetization_per_site, ghosts)

        #make sure cell is big enough for MT poisson solver
        atoms = structure_with_tags.get_ase()
        atoms.cell = 2 * (np.ptp(atoms.positions, axis=0) + 7.5)
        atoms.center()

        builder = Cp2kCalculation.get_builder()
        builder.code = self.inputs.code
        builder.structure = StructureData(ase=atoms)
        accuracy = 'lq'
        self.ctx.basis = "GW_BASIS_SET"
        self.ctx.potential = "ALL_POTENTIALS"
        if 'st' in self.inputs.gw_type.value:
            self.ctx.basis = "K_GW_BASIS"
            self.ctx.potential = "POTENTIAL"
            accuracy = 'st'
        builder.file = {
            'basis':
            SinglefileData(
                file=os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  "../..", "files/cp2k", self.ctx.basis)),
            'pseudo':
            SinglefileData(
                file=os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  "../..", "files/cp2k", self.ctx.potential)),
        }

        #UKS
        if self.inputs.multiplicity.value > 0:
            input_dict['FORCE_EVAL']['DFT']['UKS'] = '.TRUE.'
            input_dict['FORCE_EVAL']['DFT'][
                'MULTIPLICITY'] = self.inputs.multiplicity.value
        # KINDS section
        if 'hq' in self.inputs.gw_type.value:
            accuracy = 'hq'
        self.ctx.kinds_section = get_kinds_section_gw(ase_structure=atoms,
                                                      accuracy=accuracy)
        dict_merge(input_dict, self.ctx.kinds_section)

        #computational resources
        nodes, tasks_per_node, threads = get_nodes(
            atoms=atoms,
            calctype='default',
            computer=self.inputs.code.computer,
            max_nodes=48,
            uks=self.inputs.multiplicity.value > 0)
        calctype = 'gw'
        if 'ic' in self.inputs.gw_type.value:
            calctype = 'gw_ic'
        self.ctx.resources_step2 = get_nodes(
            atoms=atoms,
            calctype=calctype,
            computer=self.inputs.code.computer,
            max_nodes=2048,
            uks=self.inputs.multiplicity.value > 0)
        builder.metadata.options.resources = {
            'num_machines': nodes,
            'num_mpiprocs_per_machine': tasks_per_node,
            'num_cores_per_mpiproc': threads
        }
        #walltime
        input_dict['GLOBAL']['WALLTIME'] = self.inputs.walltime_seconds.value
        input_dict['FORCE_EVAL']['DFT']['MGRID']['CUTOFF'] = self.ctx.cutoff
        builder.metadata.options.max_wallclock_seconds = self.inputs.walltime_seconds.value

        #parser
        builder.metadata.options.parser_name = "cp2k_advanced_parser"

        #handlers

        #cp2k input dictionary
        builder.parameters = Dict(dict=input_dict)

        submitted_node = self.submit(builder)
        return ToContext(first_step=submitted_node)

    def submit_second_step(self):
        """Function to submit the second step of the workchain."""
        #load input template
        input_dict = copy.deepcopy(
            self.ctx.protocols[PROTOCOLS[self.inputs.gw_type.value][1]])

        builder = Cp2kCalculation.get_builder()
        builder.code = self.inputs.code
        builder.structure = self.ctx.first_step.inputs.structure
        builder.file = {
            'basis':
            SinglefileData(
                file=os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  "../../", "files/cp2k", self.ctx.basis)),
            'pseudo':
            SinglefileData(
                file=os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  "../..", "files/cp2k", self.ctx.potential)),
        }

        #restart from wfn of step1
        builder.parent_calc_folder = self.ctx.first_step.outputs.remote_folder

        #UKS
        if self.inputs.multiplicity.value > 0:
            input_dict['FORCE_EVAL']['DFT']['UKS'] = '.TRUE.'
            input_dict['FORCE_EVAL']['DFT'][
                'MULTIPLICITY'] = self.inputs.multiplicity.value
        # KINDS section
        dict_merge(input_dict, self.ctx.kinds_section)

        #computational resources
        nodes, tasks_per_node, threads = self.ctx.resources_step2
        builder.metadata.options.resources = {
            'num_machines': nodes,
            'num_mpiprocs_per_machine': tasks_per_node,
            'num_cores_per_mpiproc': threads
        }

        #walltime
        input_dict['GLOBAL']['WALLTIME'] = self.inputs.walltime_seconds.value
        input_dict['FORCE_EVAL']['DFT']['MGRID']['CUTOFF'] = self.ctx.cutoff
        builder.metadata.options.max_wallclock_seconds = self.inputs.walltime_seconds.value

        #parser
        builder.metadata.options.parser_name = "cp2k_advanced_parser"

        #handlers

        #cp2k input dictionary
        builder.parameters = Dict(dict=input_dict)

        submitted_node = self.submit(builder)
        return ToContext(second_step=submitted_node)

    def finalize(self):
        self.report("Finalizing...")
        #self.out('opt_structure',self.ctx.opt.outputs.output_structure)
