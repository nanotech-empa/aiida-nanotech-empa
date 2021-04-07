import os
import pathlib
import yaml
import copy
import numpy as np

from aiida.engine import WorkChain, ToContext, ExitCode
from aiida.orm import Int, Float, Str, Code, Dict, List, Bool
from aiida.orm import SinglefileData, StructureData
from aiida_nanotech_empa.workflows.cp2k.cp2k_utils import get_kinds_section, determine_kinds, dict_merge, get_nodes, get_cutoff
from aiida_cp2k.calculations import Cp2kCalculation

from aiida_nanotech_empa.utils import common_utils

ALLOWED_PROTOCOLS = ['gapw_std', 'gapw_hq', 'gpw_std']


class Cp2kMoleculeGwWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("code", valid_type=Code)

        spec.input("protocol",
                   valid_type=Str,
                   default=lambda: Str('gapw_std'),
                   required=False,
                   help="Either 'gapw_std', 'gapw_hq', 'gpw_std'")

        spec.input("image_charge",
                   valid_type=Bool,
                   default=lambda: Bool(False),
                   required=False,
                   help="Run the image charge correction calculation.")

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
                   default=lambda: Int(600),
                   required=False)
        spec.input("max_nodes",
                   valid_type=Int,
                   default=lambda: Int(2056),
                   required=False)
        spec.input("structure", valid_type=StructureData)

        spec.input("debug",
                   valid_type=Bool,
                   default=lambda: Bool(False),
                   required=False,
                   help="Run with fast parameters for debugging.")

        spec.outline(cls.setup, cls.submit_first_step, cls.submit_second_step,
                     cls.finalize)
        spec.outputs.dynamic = True

        spec.exit_code(
            381,
            "ERROR_CONVERGENCE1",
            message="SCF of the first step did not converge.",
        )
        spec.exit_code(
            382,
            "ERROR_CONVERGENCE2",
            message="SCF of the second step did not converge.",
        )
        spec.exit_code(
            383,
            "ERROR_NEGATIVE_GAP",
            message="SCF produced a negative gap.",
        )
        spec.exit_code(
            390,
            "ERROR_TERMINATION",
            message="One or more steps of the work chain failed.",
        )

    def setup(self):
        self.report("Inspecting input and setting up things")

    def submit_first_step(self):
        """Function to submit the first step of the workchain."""
        #pylint: disable=too-many-locals

        if self.inputs.protocol not in ALLOWED_PROTOCOLS:
            self.report("Error: protocol not supported.")
            return self.exit_codes.ERROR_TERMINATION

        protocol = self.inputs.protocol.value

        protocol_full = protocol + "_scf_step"

        #load input template
        with open(
                pathlib.Path(__file__).parent /
                './protocols/gw_protocols.yml') as handle:
            self.ctx.protocols = yaml.safe_load(handle)
            input_dict = copy.deepcopy(self.ctx.protocols[protocol_full])

        structure = self.inputs.structure
        self.ctx.cutoff = get_cutoff(structure=structure)
        magnetization_per_site = copy.deepcopy(
            self.inputs.magnetization_per_site)
        ghost_per_site = None

        #add ghost atoms in case of gw-ic
        if self.inputs.image_charge.value:
            atoms = self.inputs.structure.get_ase()
            image = atoms.copy()
            image.positions[:, 2] = (2 * self.inputs.z_ic_plane.value -
                                     atoms.positions[:, 2])
            ghost_per_site = [0 for a in atoms] + [1 for a in image]
            if (magnetization_per_site):
                magnetization_per_site += [0 for i in range(len(image))]
            structure = StructureData(ase=atoms + image)

        structure_with_tags, kinds_dict = determine_kinds(
            structure, magnetization_per_site, ghost_per_site)

        #make sure cell is big enough for MT poisson solver
        if self.inputs.debug:
            extra_cell = 5.0
        else:
            extra_cell = 15.0
        self.ctx.atoms = structure_with_tags.get_ase()
        self.ctx.atoms.cell = 2 * (np.ptp(self.ctx.atoms.positions,
                                          axis=0)) + extra_cell
        self.ctx.atoms.center()

        builder = Cp2kCalculation.get_builder()
        builder.code = self.inputs.code
        builder.structure = StructureData(ase=self.ctx.atoms)

        if protocol in ['gapw_std', 'gapw_hq']:
            self.ctx.basis = "GW_BASIS_SET"
            self.ctx.potential = "ALL_POTENTIALS"
        elif protocol == 'gpw_std':
            self.ctx.basis = "K_GW_BASIS"
            self.ctx.potential = "POTENTIAL"

        builder.file = {
            'basis':
            SinglefileData(
                file=os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  ".", "data", self.ctx.basis)),
            'pseudo':
            SinglefileData(
                file=os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  ".", "data", self.ctx.potential)),
        }

        #UKS
        if self.inputs.multiplicity.value > 0:
            input_dict['FORCE_EVAL']['DFT']['UKS'] = '.TRUE.'
            input_dict['FORCE_EVAL']['DFT'][
                'MULTIPLICITY'] = self.inputs.multiplicity.value

        # KINDS section
        self.ctx.kinds_section = get_kinds_section(kinds_dict,
                                                   protocol=protocol)
        dict_merge(input_dict, self.ctx.kinds_section)

        #computational resources
        nodes, tasks_per_node, threads = get_nodes(
            atoms=self.ctx.atoms,
            calctype='default',
            computer=self.inputs.code.computer,
            max_nodes=min(48, self.inputs.max_nodes.value),
            uks=self.inputs.multiplicity.value > 0)

        builder.metadata.options.resources = {
            'num_machines': nodes,
            'num_mpiprocs_per_machine': tasks_per_node,
            'num_cores_per_mpiproc': threads
        }
        #walltime
        input_dict['GLOBAL']['WALLTIME'] = max(
            self.inputs.walltime_seconds.value - 600, 600)
        builder.metadata.options.max_wallclock_seconds = self.inputs.walltime_seconds.value

        #cutoff
        input_dict['FORCE_EVAL']['DFT']['MGRID']['CUTOFF'] = self.ctx.cutoff

        if self.inputs.debug:
            input_dict['FORCE_EVAL']['DFT']['PRINT']['MO_CUBES'][
                'STRIDE'] = '6 6 6'
            input_dict['FORCE_EVAL']['DFT']['PRINT']['E_DENSITY_CUBE'][
                'STRIDE'] = '6 6 6'

        #parser
        builder.metadata.options.parser_name = "cp2k_advanced_parser"

        #handlers

        #cp2k input dictionary
        builder.parameters = Dict(dict=input_dict)

        submitted_node = self.submit(builder)
        return ToContext(first_step=submitted_node)

    def submit_second_step(self):
        """Function to submit the second step of the workchain."""

        if not common_utils.check_if_calc_ok(self, self.ctx.first_step):
            return self.exit_codes.ERROR_TERMINATION

        scf_out_params = self.ctx.first_step.outputs.output_parameters

        if not scf_out_params['motion_step_info']['scf_converged'][-1]:
            self.report("SCF step did not converge")
            return self.exit_codes.ERROR_CONVERGENCE1

        if min(scf_out_params['bandgap_spin1_au'],
               scf_out_params['bandgap_spin2_au']) < 0.0:
            self.report("Negative gap!")
            return self.exit_codes.ERROR_NEGATIVE_GAP

        protocol = self.inputs.protocol.value

        if self.inputs.image_charge.value:
            protocol_full = protocol + "_ic_step"
        else:
            protocol_full = protocol + "_gw_step"

        #load input template
        input_dict = copy.deepcopy(self.ctx.protocols[protocol_full])

        builder = Cp2kCalculation.get_builder()
        builder.code = self.inputs.code
        builder.structure = self.ctx.first_step.inputs.structure
        builder.file = {
            'basis':
            SinglefileData(
                file=os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  ".", "data", self.ctx.basis)),
            'pseudo':
            SinglefileData(
                file=os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  ".", "data", self.ctx.potential)),
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
        nodes, tasks_per_node, threads = get_nodes(
            atoms=self.ctx.atoms,
            calctype='gw_ic' if self.inputs.image_charge.value else 'gw',
            computer=self.inputs.code.computer,
            max_nodes=min(2048, self.inputs.max_nodes.value),
            uks=self.inputs.multiplicity.value > 0)

        builder.metadata.options.resources = {
            'num_machines': nodes,
            'num_mpiprocs_per_machine': tasks_per_node,
            'num_cores_per_mpiproc': threads
        }

        #walltime
        input_dict['GLOBAL']['WALLTIME'] = max(
            self.inputs.walltime_seconds.value - 600, 600)
        input_dict['FORCE_EVAL']['DFT']['MGRID']['CUTOFF'] = self.ctx.cutoff
        builder.metadata.options.max_wallclock_seconds = self.inputs.walltime_seconds.value

        if self.inputs.debug:
            input_dict['FORCE_EVAL']['DFT']['PRINT']['MO_CUBES'][
                'STRIDE'] = '6 6 6'
            input_dict['FORCE_EVAL']['DFT']['PRINT']['E_DENSITY_CUBE'][
                'STRIDE'] = '6 6 6'

        #parser
        builder.metadata.options.parser_name = "nanotech_empa.cp2k_gw_parser"

        #handlers

        #cp2k input dictionary
        builder.parameters = Dict(dict=input_dict)

        submitted_node = self.submit(builder)
        return ToContext(second_step=submitted_node)

    def finalize(self):
        self.report("Finalizing...")

        if not common_utils.check_if_calc_ok(self, self.ctx.second_step):
            return self.exit_codes.ERROR_TERMINATION
        if not self.ctx.second_step.outputs.std_output_parameters[
                'motion_step_info']['scf_converged'][-1]:
            self.report("GW step did not converge")
            return self.exit_codes.ERROR_CONVERGENCE2

        self.out('std_output_parameters',
                 self.ctx.second_step.outputs.std_output_parameters)
        self.out('gw_output_parameters',
                 self.ctx.second_step.outputs.gw_output_parameters)

        return ExitCode(0)
