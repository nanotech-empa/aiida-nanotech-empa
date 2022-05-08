import os
import pathlib
import yaml
import copy

from aiida.engine import WorkChain, ToContext, ExitCode, calcfunction
from aiida.orm import Int, Bool, Code, Dict, List, Str
from aiida.orm import SinglefileData, StructureData
from aiida.plugins import WorkflowFactory
from aiida_nanotech_empa.workflows.cp2k.cp2k_utils import get_kinds_section, determine_kinds, dict_merge, get_nodes, get_cutoff
from aiida_nanotech_empa.utils import analyze_structure

from aiida_nanotech_empa.utils import common_utils

Cp2kBaseWorkChain = WorkflowFactory('cp2k.base')


@calcfunction
def split_structure(charges, fragments, multiplicity):
    charges = list(charges)
    multiplicity = list(multiplicity)
    fragments = list(fragments)
    if not charges:
        charges = [0 for f in fragments]
    if not multiplicity:
        multiplicity = [0 for f in fragments]
    setup_dict = {}
    for i, fragment in enumerate(fragments):
        label = 'frag_' + str(i)
        fragment.sort()
        setup_dict[label] = {
            'charge': charges[i],
            'multiplicity': multiplicity[i],
            'fragment': analyze_structure.list_to_string_range(fragment,
                                                               shift=1)
        }
    return setup_dict


class Cp2kBsseWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("code", valid_type=Code)
        spec.input("structure", valid_type=StructureData)

        spec.input(
            "charge",  # +1 means one electron removed
            valid_type=List,
            default=lambda: List(list=[]),
            required=False)
        spec.input("multiplicity",
                   valid_type=List,
                   default=lambda: List(list=[]),
                   required=False)
        spec.input("whole_multiplicity",
                   valid_type=Int,
                   default=lambda: Int(0),
                   required=False)
        spec.input("magnetization_per_site",
                   valid_type=List,
                   default=lambda: List(list=[]),
                   required=False)
        spec.input("fragments",
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
        spec.input("max_nodes",
                   valid_type=Int,
                   default=lambda: Int(48),
                   required=False)
        spec.input("walltime_seconds",
                   valid_type=Int,
                   default=lambda: Int(7200),
                   required=False)

        #workchain outline
        spec.outline(cls.setup, cls.submit_calc, cls.finalize)
        spec.outputs.dynamic = True

        spec.exit_code(
            390,
            "ERROR_TERMINATION",
            message="One or more steps of the work chain failed.",
        )
        spec.exit_code(
            400,
            "ERROR_NFRAGMENTS",
            message="Only the case of 2 fragments is supported.",
        )

    def setup(self):
        self.report("Inspecting input and setting up things")
        self.ctx.fragments = split_structure(self.inputs.charge,
                                             self.inputs.fragments,
                                             self.inputs.multiplicity)
        if len(list(self.inputs.fragments)) != 2:
            return self.exit_codes.ERROR_NFRAGMENTS
        return ExitCode(0)

        # --------------------------------------------------

    # pylint: disable=too-many-locals
    def submit_calc(self):
        self.report("Submitting BSSE")
        with open(pathlib.Path(__file__).parent /
                  './protocols/bsse_protocol.yml',
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

        ase_atoms = structure_with_tags.get_ase()

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

        # vdw
        if not self.inputs.vdw.value:
            input_dict['FORCE_EVAL']['DFT']['XC'].pop('VDW_POTENTIAL')

        #UKS
        if not all(m == 0 for m in self.inputs.multiplicity):
            input_dict['FORCE_EVAL']['DFT']['UKS'] = '.TRUE.'
            #input_dict['FORCE_EVAL']['DFT'][
            #    'MULTIPLICITY'] = self.inputs.whole_multiplicity.value

        #cutoff
        input_dict['FORCE_EVAL']['DFT']['MGRID']['CUTOFF'] = self.ctx.cutoff

        # KINDS section
        self.ctx.kinds_section = get_kinds_section(kinds_dict,
                                                   protocol='gpw',
                                                   bsse=True)
        dict_merge(input_dict, self.ctx.kinds_section)

        # BSSE section
        frag = 'FRAGMENT'
        conf = 'CONFIGURATION'
        fragments = self.ctx.fragments
        input_dict['FORCE_EVAL']['BSSE'] = {}
        for details in fragments.items():
            #    input_dict['FORCE_EVAL']['BSSE'][conf] = {
            #        'CHARGE': details[1]['charge'],
            #        'MULTIPLICITY': details[1]['multiplicity'],
            #        'GLB_CONF' : 1
            #    }
            input_dict['FORCE_EVAL']['BSSE'][frag] = {
                'LIST': details[1]['fragment']
            }
            frag = frag + ' '
        #    conf = conf + ' '

        ##check the following with BSSE experts, it's not all combinations....
        input_dict['FORCE_EVAL']['BSSE'][conf] = {
            'CHARGE': fragments['frag_0']['charge'],
            'MULTIPLICITY': fragments['frag_0']['multiplicity'],
            'GLB_CONF': '1 0',
            'SUB_CONF': '1 0'
        }
        conf += ' '
        input_dict['FORCE_EVAL']['BSSE'][conf] = {
            'CHARGE': fragments['frag_0']['charge'],
            'MULTIPLICITY': fragments['frag_0']['multiplicity'],
            'GLB_CONF': '1 1',
            'SUB_CONF': '1 0'
        }
        conf += ' '
        input_dict['FORCE_EVAL']['BSSE'][conf] = {
            'CHARGE': fragments['frag_1']['charge'],
            'MULTIPLICITY': fragments['frag_1']['multiplicity'],
            'GLB_CONF': '1 1',
            'SUB_CONF': '0 1'
        }
        conf += ' '
        input_dict['FORCE_EVAL']['BSSE'][conf] = {
            'CHARGE':
            fragments['frag_0']['charge'] + fragments['frag_1']['charge'],
            'MULTIPLICITY': self.inputs.whole_multiplicity,
            'GLB_CONF': '1 1',
            'SUB_CONF': '1 1'
        }

        #computational resources
        max_nodes = self.inputs.max_nodes.value
        if self.inputs.protocol.value == 'debug':
            max_nodes = 1
        nodes, tasks_per_node, threads = get_nodes(
            atoms=ase_atoms,
            calctype='slab',
            computer=self.inputs.code.computer,
            max_nodes=max_nodes,
            uks=self.inputs.whole_multiplicity.value > 0)

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

        #cp2k input dictionary
        builder.cp2k.parameters = Dict(dict=input_dict)

        submitted_node = self.submit(builder)
        return ToContext(opt=submitted_node)

    def finalize(self):
        self.report("Finalizing...")

        if not common_utils.check_if_calc_ok(self, self.ctx.opt):
            return self.exit_codes.ERROR_TERMINATION

        for out in self.ctx.opt.outputs:
            self.out(out, self.ctx.opt.outputs[out])

        return ExitCode(0)
