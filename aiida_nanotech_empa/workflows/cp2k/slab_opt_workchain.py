import os
import pathlib
import yaml
import copy

from aiida.engine import WorkChain, ToContext, ExitCode
from aiida.orm import Int, Bool, Code, Dict, List, Str
from aiida.orm import SinglefileData, StructureData
from aiida.plugins import WorkflowFactory
from aiida_nanotech_empa.workflows.cp2k.cp2k_utils import get_kinds_section, determine_kinds, dict_merge, get_nodes, get_cutoff

from aiida_nanotech_empa.utils import common_utils, analyze_structure

Cp2kBaseWorkChain = WorkflowFactory('cp2k.base')


class Cp2kSlabOptWorkChain(WorkChain):
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
        spec.input("fixed_atoms",
                   valid_type=Str,
                   default=lambda: Str(''),
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
        spec.input("max_nodes",
                   valid_type=Int,
                   default=lambda: Int(48),
                   required=False)
        spec.input("walltime_seconds",
                   valid_type=Int,
                   default=lambda: Int(7200),
                   required=False)
        spec.input("debug",
                   valid_type=Bool,
                   default=lambda: Bool(False),
                   required=False,
                   help="Run with fast parameters for debugging.")

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
                  './protocols/slab_opt_protocol.yml',
                  encoding='utf-8') as handle:
            protocols = yaml.safe_load(handle)
            input_dict = copy.deepcopy(protocols['default'])

        structure = self.inputs.structure
        #cutoff
        self.ctx.cutoff = get_cutoff(structure=structure)

        #get initial magnetization
        magnetization_per_site = copy.deepcopy(
            self.inputs.magnetization_per_site)
        structure_with_tags, kinds_dict = determine_kinds(
            structure, magnetization_per_site)

        self.ctx.atoms = structure_with_tags.get_ase()

        builder = Cp2kBaseWorkChain.get_builder()
        builder.cp2k.code = self.inputs.code
        builder.cp2k.structure = StructureData(ase=self.ctx.atoms)
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

        #fixed atoms
        input_dict['MOTION']['CONSTRAINT']['FIXED_ATOMS'][
            'LIST'] = self.inputs.fixed_atoms.value

        #cutoff
        input_dict['FORCE_EVAL']['DFT']['MGRID']['CUTOFF'] = self.ctx.cutoff

        if self.inputs.debug:
            input_dict['MOTION']['GEO_OPT']['MAX_FORCE'] = 0.1
            input_dict['MOTION']['GEO_OPT']['RMS_DR'] = 0.1
            input_dict['MOTION']['GEO_OPT']['RMS_FORCE'] = 0.1
            input_dict['MOTION']['GEO_OPT']['MAX_DR'] = 0.1
            input_dict['FORCE_EVAL']['DFT']['SCF']['EPS_SCF'] = 1e-4
            input_dict['FORCE_EVAL']['DFT']['SCF']['OUTER_SCF'][
                'EPS_SCF'] = 1e-4

        # KINDS section
        self.ctx.kinds_section = get_kinds_section(kinds_dict, protocol='gpw')
        dict_merge(input_dict, self.ctx.kinds_section)

        #computational resources
        nodes, tasks_per_node, threads = get_nodes(
            atoms=self.ctx.atoms,
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

        submitted_node = self.submit(builder)
        return ToContext(opt=submitted_node)

    def finalize(self):
        self.report("Finalizing...")

        if not common_utils.check_if_calc_ok(self, self.ctx.opt):
            return self.exit_codes.ERROR_TERMINATION

        for out in self.ctx.opt.outputs:
            self.out(out, self.ctx.opt.outputs[out])

        # Add extras
        struc = self.ctx.opt.outputs.output_structure
        ase_geom = struc.get_ase()
        self.node.set_extra('thumbnail',
                            common_utils.thumbnail(ase_struc=ase_geom))

        # add formula to extra as molecule@surface
        try:  #mainly for debug cases where the analyzer could crash due to odd geometries
            analyzer = analyze_structure.StructureAnalyzer()
            analyzer.structure = ase_geom
            res = analyzer.details

            mol_formula = ''
            for imol in res['all_molecules']:
                mol_formula += ase_geom[imol].get_chemical_formula() + ' '
            if len(res['slabatoms']) > 0:
                mol_formula += 'at ' + ase_geom[
                    res['slabatoms']].get_chemical_formula()
                if len(res['bottom_H']) > 0:
                    mol_formula += ' saturated: ' + ase_geom[
                        res['bottom_H']].get_chemical_formula()
                if len(res['adatoms']) > 0:
                    mol_formula += ' Adatoms: ' + ase_geom[
                        res['adatoms']].get_chemical_formula()
        except ValueError:
            mol_formula = struc.get_formula()

        self.node.set_extra('formula', mol_formula)

        return ExitCode(0)
