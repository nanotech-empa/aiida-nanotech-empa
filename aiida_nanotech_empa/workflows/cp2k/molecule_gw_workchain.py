import os
import pathlib
import yaml
import copy
import numpy as np

from aiida.engine import WorkChain, ToContext, ExitCode, while_
from aiida.orm import Int, Float, Str, Code, Dict, List, Bool
from aiida.orm import SinglefileData, StructureData

from aiida_nanotech_empa.workflows.cp2k.cp2k_utils import get_kinds_section, determine_kinds, dict_merge, get_nodes, get_cutoff
from aiida_nanotech_empa.utils import common_utils

from aiida_cp2k.calculations import Cp2kCalculation

ALLOWED_PROTOCOLS = ['gapw_std', 'gapw_hq', 'gpw_std']


class Cp2kMoleculeGwWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("code", valid_type=Code)
        spec.input("structure", valid_type=StructureData)

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
        spec.input("debug",
                   valid_type=Bool,
                   default=lambda: Bool(False),
                   required=False,
                   help="Run with fast parameters for debugging.")

        spec.input(
            'options',
            valid_type=Dict,
            required=False,
            help=
            "User-defined metadata.options that override the automatic ones.")

        spec.outline(
            cls.setup,
            while_(cls.scf_is_not_done)(cls.submit_scf, cls.check_scf),
            cls.submit_gw, cls.finalize)
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

        if self.inputs.protocol not in ALLOWED_PROTOCOLS:
            self.report("Error: protocol not supported.")
            return self.exit_codes.ERROR_TERMINATION

        # Load protocol templates
        with open(
                pathlib.Path(__file__).parent.joinpath(
                    './protocols/gw_protocols.yml')) as handle:
            self.ctx.protocols = yaml.safe_load(handle)

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

        # KINDS section
        self.ctx.kinds_section = get_kinds_section(
            kinds_dict, protocol=self.inputs.protocol)

        #make sure cell is big enough for MT poisson solver
        if self.inputs.debug:
            extra_cell = 5.0
        else:
            extra_cell = 15.0
        atoms = structure_with_tags.get_ase()
        atoms.cell = 2 * (np.ptp(atoms.positions, axis=0)) + extra_cell
        atoms.center()
        self.ctx.structure = StructureData(ase=atoms)

        # --------------------------------------------------
        # Determine which basis and pseudo files to include
        if self.inputs.protocol in ['gapw_std', 'gapw_hq']:
            basis = "GW_BASIS_SET"
            potential = "ALL_POTENTIALS"
        elif self.inputs.protocol == 'gpw_std':
            basis = "K_GW_BASIS"
            potential = "POTENTIAL"

        self.ctx.files = {
            'basis':
            SinglefileData(
                file=os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  ".", "data", basis)),
            'pseudo':
            SinglefileData(
                file=os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  ".", "data", potential)),
        }
        # --------------------------------------------------

        self.ctx.current_scf_protocol = None
        self.ctx.scf_restart_from_last = False

        return ExitCode(0)

    def scf_is_not_done(self):

        if hasattr(self.ctx, 'scf'):
            scf_out_params = self.ctx.scf.outputs.output_parameters

            scf_converged = scf_out_params['motion_step_info'][
                'scf_converged'][-1]

            gap_positive = min(scf_out_params['bandgap_spin1_au'],
                               scf_out_params['bandgap_spin2_au']) >= 0.0

            if scf_converged and gap_positive:
                self.report("SCF finished well, continue to GW!")
                return False

            if scf_converged:
                # If the SCF converged but the gap was negative, restart in the next step
                self.ctx.scf_restart_from_last = True
            else:
                self.ctx.scf_restart_from_last = False

            self.report(
                "Something went wrong in the SCF, try the next protocol.")

        return True

    def _check_and_set_uks(self, input_dict):
        if self.inputs.multiplicity.value > 0:
            input_dict['FORCE_EVAL']['DFT']['UKS'] = '.TRUE.'
            input_dict['FORCE_EVAL']['DFT'][
                'MULTIPLICITY'] = self.inputs.multiplicity.value

    def _set_debug(self, input_dict):
        input_dict['FORCE_EVAL']['DFT']['PRINT']['MO_CUBES'][
            'STRIDE'] = '6 6 6'
        input_dict['FORCE_EVAL']['DFT']['PRINT']['E_DENSITY_CUBE'][
            'STRIDE'] = '6 6 6'

    def _get_resources(self, calctype, max_nodes_cap):
        nodes, tasks_per_node, threads = get_nodes(
            atoms=self.ctx.structure.get_ase(),
            calctype=calctype,
            computer=self.inputs.code.computer,
            max_nodes=min(max_nodes_cap, self.inputs.max_nodes.value),
            uks=self.inputs.multiplicity.value > 0)

        res = {
            'num_machines': nodes,
            'num_mpiprocs_per_machine': tasks_per_node,
            'num_cores_per_mpiproc': threads
        }

        return res

    def _get_metadata_options(self, calctype, max_nodes_cap):
        # automatically determined metadata_options
        options = {
            'resources': self._get_resources(calctype, max_nodes_cap),
            'max_wallclock_seconds': self.inputs.walltime_seconds.value,
        }
        # If user specified any, overwrite those:
        if 'options' in self.inputs:
            dict_merge(options, dict(self.inputs.options))
        return options

    def submit_scf(self):

        # Try the next SCF section:
        if self.ctx.current_scf_protocol is None:
            # First try
            self.ctx.current_scf_protocol = 'scf_ot_cg'
        elif self.ctx.current_scf_protocol == 'scf_ot_cg':
            # Second try
            self.ctx.current_scf_protocol = 'scf_ot_diis'
        elif self.ctx.current_scf_protocol == 'scf_ot_diis':
            # Third try
            self.ctx.current_scf_protocol = 'scf_diag_smearing'
        else:
            # Failure
            return self.exit_codes.ERROR_CONVERGENCE1

        self.report(
            f"Submitting SCF (protocol {self.ctx.current_scf_protocol})")

        # -------------------------------------------------------
        # Build the input dictionary

        step_protocol = self.inputs.protocol.value + "_scf_step"
        input_dict = copy.deepcopy(self.ctx.protocols[step_protocol])

        scf_section = copy.deepcopy(
            self.ctx.protocols[self.ctx.current_scf_protocol])
        input_dict['FORCE_EVAL']['DFT']['SCF'] = scf_section

        self._check_and_set_uks(input_dict)

        dict_merge(input_dict, self.ctx.kinds_section)

        input_dict['GLOBAL']['WALLTIME'] = max(
            self.inputs.walltime_seconds.value - 600, 600)

        input_dict['FORCE_EVAL']['DFT']['MGRID']['CUTOFF'] = self.ctx.cutoff

        if self.inputs.debug:
            self._set_debug(input_dict)

        # -------------------------------------------------------

        builder = Cp2kCalculation.get_builder()
        builder.code = self.inputs.code
        builder.structure = self.ctx.structure
        builder.file = self.ctx.files

        if hasattr(self.ctx, 'scf') and self.ctx.scf_restart_from_last:
            builder.parent_calc_folder = self.ctx.scf.outputs.remote_folder
            input_dict['FORCE_EVAL']['DFT'][
                'RESTART_FILE_NAME'] = './parent_calc/aiida-RESTART.wfn'

        builder.parameters = Dict(dict=input_dict)

        builder.metadata.options = self._get_metadata_options(
            calctype='default', max_nodes_cap=48)
        builder.metadata.options['parser_name'] = "cp2k_advanced_parser"

        submitted_node = self.submit(builder)
        return ToContext(scf=submitted_node)

    def check_scf(self):
        if not common_utils.check_if_calc_ok(self, self.ctx.scf):
            return self.exit_codes.ERROR_TERMINATION
        return ExitCode(0)

    def submit_gw(self):

        self.report("Submitting GW.")

        # -------------------------------------------------------
        # Build the input dictionary

        if self.inputs.image_charge.value:
            step_protocol = self.inputs.protocol.value + "_ic_step"
        else:
            step_protocol = self.inputs.protocol.value + "_gw_step"

        input_dict = copy.deepcopy(self.ctx.protocols[step_protocol])

        scf_section = copy.deepcopy(
            self.ctx.protocols[self.ctx.current_scf_protocol])
        input_dict['FORCE_EVAL']['DFT']['SCF'] = scf_section

        self._check_and_set_uks(input_dict)

        dict_merge(input_dict, self.ctx.kinds_section)

        input_dict['GLOBAL']['WALLTIME'] = max(
            self.inputs.walltime_seconds.value - 600, 600)

        input_dict['FORCE_EVAL']['DFT']['MGRID']['CUTOFF'] = self.ctx.cutoff

        if self.inputs.debug:
            self._set_debug(input_dict)

        # -------------------------------------------------------

        builder = Cp2kCalculation.get_builder()
        builder.code = self.inputs.code
        builder.structure = self.ctx.structure
        builder.file = self.ctx.files

        #restart from wfn of step1
        builder.parent_calc_folder = self.ctx.scf.outputs.remote_folder

        builder.metadata.options = self._get_metadata_options(
            calctype='gw_ic' if self.inputs.image_charge.value else 'gw',
            max_nodes_cap=2048)
        builder.metadata.options[
            'parser_name'] = "nanotech_empa.cp2k_gw_parser"

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
