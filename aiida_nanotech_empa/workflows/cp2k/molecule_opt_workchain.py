import os
import pathlib
import yaml
import copy
import numpy as np

from aiida.engine import WorkChain, ToContext, ExitCode
from aiida.orm import Int, Bool, Code, Dict, List, Str
from aiida.orm import SinglefileData, StructureData
from aiida.plugins import WorkflowFactory
from aiida_nanotech_empa.workflows.cp2k.cp2k_utils import get_kinds_section, determine_kinds, dict_merge, get_cutoff

from aiida_nanotech_empa.utils import common_utils

Cp2kBaseWorkChain = WorkflowFactory('cp2k.base')


class Cp2kMoleculeOptWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("code", valid_type=Code)
        spec.input("structure", valid_type=StructureData)

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
        spec.input(
            "options",
            valid_type=dict,
            non_db=True,
            required=False,
            help=
            "Define options for the cacluations: walltime, memory, CPUs, etc.")

        #workchain outline
        spec.outline(cls.setup, cls.submit_calc, cls.finalize)
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
        with open(pathlib.Path(__file__).parent /
                  './protocols/molecule_opt_protocol.yml',
                  encoding='utf-8') as handle:
            protocols = yaml.safe_load(handle)
            input_dict = copy.deepcopy(protocols[self.inputs.protocol.value])

        structure = self.inputs.structure
        #cutoff
        self.ctx.cutoff = get_cutoff(structure=structure)

        #get initial magnetization
        magnetization_per_site = copy.deepcopy(
            self.inputs.magnetization_per_site)
        structure_with_tags, kinds_dict = determine_kinds(
            structure, magnetization_per_site)

        #make sure cell is big enough for MT poisson solver
        if self.inputs.protocol.value == "debug":
            extra_cell = 5.0
        else:
            extra_cell = 15.0
        ase_atoms = structure_with_tags.get_ase()
        ase_atoms.cell = 2 * (np.ptp(ase_atoms.positions, axis=0)) + extra_cell
        ase_atoms.center()

        builder = Cp2kBaseWorkChain.get_builder()
        builder.cp2k.code = self.inputs.code
        builder.cp2k.structure = StructureData(ase=ase_atoms)
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
        input_dict['FORCE_EVAL']['DFT']['CHARGE'] = self.inputs.charge.value

        # vdw
        if not self.inputs.vdw.value:
            input_dict['FORCE_EVAL']['DFT']['XC'].pop('VDW_POTENTIAL')
        else:
            builder.cp2k.file['dftd3'] = SinglefileData(
                file=os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  ".", "data", "dftd3.dat"))

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

        # Setup options.
        if 'options' in self.inputs:
            builder.cp2k.metadata.options = self.inputs.options

        # Setup walltime.
        if 'max_wallclock_seconds' in self.inputs.options:
            input_dict['GLOBAL']['WALLTIME'] = max(
                self.inputs.options['max_wallclock_seconds'] - 600, 600)

        #parser
        builder.cp2k.metadata.options.parser_name = "cp2k_advanced_parser"

        #handlers
        builder.handler_overrides = Dict(
            dict={'restart_incomplete_calculation': True})

        #cp2k input dictionary
        builder.cp2k.parameters = Dict(dict=input_dict)

        submitted_node = self.submit(builder)
        return ToContext(opt=submitted_node)

    def finalize(self):
        self.report("Finalizing...")

        if not self.ctx.opt.is_finished_ok:
            return self.exit_codes.ERROR_TERMINATION

        for out in self.ctx.opt.outputs:
            self.out(out, self.ctx.opt.outputs[out])

        # Add extras
        struc = self.ctx.opt.outputs.output_structure
        self.node.set_extra('thumbnail',
                            common_utils.thumbnail(ase_struc=struc.get_ase()))
        self.node.set_extra('formula', struc.get_formula())

        return ExitCode(0)
