import copy
import os
import pathlib

import numpy as np
import yaml

from aiida.engine import WorkChain
from aiida.orm import Code, Dict, List, SinglefileData, Str, StructureData
from aiida.orm import SinglefileData

from aiida.plugins import WorkflowFactory

from aiida_nanotech_empa.utils import common_utils
from aiida_nanotech_empa.workflows.cp2k.cp2k_utils import (
    determine_kinds,
    dict_merge,
    get_cutoff,
    get_kinds_section,
)

Cp2kBaseWorkChain = WorkflowFactory("cp2k.base")


class Cp2kDiagWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super(Cp2kDiagWorkChain, cls).define(spec)

        spec.input("cp2k_code", valid_type=Code)
        spec.input("structure", valid_type=StructureData)
        spec.input("wfn_file_path", valid_type=Str, required=False)
        spec.input("dft_params", valid_type=Dict, default=lambda: Dict(dict={}))
        spec.input("pdos_lists", valid_type=List, required=False)
        spec.input("settings", valid_type=Dict, required=False)
        spec.input(
            "options",
            valid_type=Dict,
            default=lambda: Dict(
                dict={
                    "max_wallclock_seconds": 600,
                    "resources": {
                        "num_machines": 1,
                        "num_mpiprocs_per_machine": 1,
                        "num_cores_per_mpiproc": 1,
                    },
                }
            ),
            required=False,
            help="Define options for the cacluations: walltime, memory, CPUs, etc.",
        )
        spec.outline(
            cls.setup,
            cls.run_ot_scf,
            cls.run_diag_scf,
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

        # set up mol UKS parameters

        self.ctx.dft_params = copy.deepcopy(self.inputs.dft_params.get_dict())
        # resources
        self.ctx.options = self.inputs.options.get_dict()
        if self.ctx.dft_params["protocol"] == "debug":
            self.ctx.options = {
                "max_wallclock_seconds": 600,
                "resources": {
                    "num_machines": 1,
                    "num_mpiprocs_per_machine": 1,
                    "num_cores_per_mpiproc": 1,
                },
            }

        if not self.ctx.dft_params["uks"]:
            self.ctx.dft_params["spin_up_guess"] = []
            self.ctx.dft_params["spin_dw_guess"] = []

        # get cutoff
        self.ctx.cutoff = get_cutoff(structure=structure)
        # overwrite cutoff if given in dft_params
        if "cutoff" in self.ctx.dft_params:
            self.ctx.cutoff = self.ctx.dft_params["cutoff"]

        # get initial magnetization
        spin_up_guess = self.ctx.dft_params["spin_up_guess"]
        spin_dw_guess = self.ctx.dft_params["spin_dw_guess"]
        magnetization_per_site = [
            1 if i in spin_up_guess else -1 if i in spin_dw_guess else 0
            for i in range(self.ctx.n_atoms)
        ]
        structure_with_tags, kinds_dict = determine_kinds(
            structure, magnetization_per_site
        )

        ase_atoms = structure_with_tags.get_ase()
        # PERIODIC: only NONE and XYZ are supported
        if self.ctx.dft_params["periodic"] == "NONE":
            # make sure cell is big enough for MT poisson solver and center positions
            if self.ctx.dft_params["protocol"] == "debug":
                extra_cell = 9.0  # angstrom
            else:
                extra_cell = 15.0
            ase_atoms.cell = 2 * (np.ptp(ase_atoms.positions, axis=0)) + extra_cell
            ase_atoms.center()

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
            input_dict = copy.deepcopy(protocols[self.ctx.dft_params["protocol"]])

        builder = Cp2kBaseWorkChain.get_builder()
        builder.cp2k.code = self.inputs.cp2k_code
        builder.cp2k.structure = StructureData(ase=self.ctx.structure_with_tags)

        builder.cp2k.file = self.ctx.files
        if "wfn_file_path" in self.inputs:
            builder.cp2k.parent_calc_folder = self.inputs.wfn_file_path.value

        if "charge" in self.ctx.dft_params:
            input_dict["FORCE_EVAL"]["DFT"]["CHARGE"] = self.ctx.dft_params["charge"]
        input_dict["FORCE_EVAL"]["DFT"]["XC"].pop("VDW_POTENTIAL")

        # POISSON_SOLVER
        if self.ctx.dft_params["periodic"] == "NONE":
            input_dict["FORCE_EVAL"]["DFT"]["POISSON"]["PERIODIC"] = "NONE"
            input_dict["FORCE_EVAL"]["DFT"]["POISSON"]["POISSON_SOLVER"] = "MT"

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

        builder.cp2k.metadata.options = self.ctx.options

        # parser
        builder.cp2k.metadata.options.parser_name = "cp2k_advanced_parser"

        # cp2k input dictionary
        builder.cp2k.parameters = Dict(input_dict)
        self.ctx.input_dict = copy.deepcopy(input_dict)

        future = self.submit(builder)
        self.to_context(ot_scf=future)

    def run_diag_scf(self):
        self.report("Running CP2K diagonalization SCF")
        if not common_utils.check_if_calc_ok(self, self.ctx.ot_scf):
            return self.exit_codes.ERROR_TERMINATION  # pylint: disable=no-member

        # load input template
        with open(
            pathlib.Path(__file__).parent / "./protocols/scf_diag_protocol.yml",
            encoding="utf-8",
        ) as handle:
            protocols = yaml.safe_load(handle)
            scf_dict = copy.deepcopy(protocols[self.ctx.dft_params["protocol"]])

        input_dict = copy.deepcopy(self.ctx.input_dict)
        if self.ctx.dft_params["elpa_switch"]:
            input_dict["GLOBAL"]["PREFERRED_DIAG_LIBRARY"] = "ELPA"
            input_dict["GLOBAL"]["ELPA_KERNEL"] = "AUTO"
            input_dict["GLOBAL"]["DBCSR"] = {"USE_MPI_ALLOCATOR": ".FALSE."}
        input_dict["FORCE_EVAL"]["DFT"].pop("SCF")
        input_dict["FORCE_EVAL"]["DFT"]["SCF"] = scf_dict
        if "added_mos" in self.ctx.dft_params:
            input_dict["FORCE_EVAL"]["DFT"]["SCF"]["ADDED_MOS"] = self.ctx.dft_params[
                "added_mos"
            ]

        # pdos
        if "pdos_lists" in self.inputs:
            pdos_list_dicts = [
                {"COMPONENTS": "", "LIST": e} for e in self.inputs.pdos_lists
            ]
            input_dict["FORCE_EVAL"]["DFT"]["PRINT"]["PDOS"] = {
                "NLUMO": self.ctx.dft_params["added_mos"],
                "LDOS": pdos_list_dicts,
            }

        smearing = "smear_t" in self.ctx.dft_params
        if smearing and self.ctx.dft_params["sc_diag"]:
            input_dict["FORCE_EVAL"]["DFT"]["SCF"]["SMEAR"][
                "ELECTRONIC_TEMPERATURE"
            ] = self.ctx.dft_params["smear_t"]
        else:
            input_dict["FORCE_EVAL"]["DFT"]["SCF"].pop("SMEAR")

        # UKS
        if (
            self.ctx.dft_params["uks"]
            and smearing
            and self.ctx.dft_params["force_multiplicity"]
        ):
            input_dict["FORCE_EVAL"]["DFT"]["SCF"]["SMEAR"]["FIXED_MAGNETIC_MOMENT"] = (
                self.ctx.dft_params["multiplicity"] - 1
            )
        # no self consistent diag
        if not self.ctx.dft_params["sc_diag"]:
            if (
                "SMEAR" in input_dict["FORCE_EVAL"]["DFT"]["SCF"]
            ):  # could have been already removed if smear false and sc_diag true
                input_dict["FORCE_EVAL"]["DFT"]["SCF"].pop("SMEAR")
            input_dict["FORCE_EVAL"]["DFT"]["SCF"]["EPS_SCF"] = "1.0E-1"
            input_dict["FORCE_EVAL"]["DFT"]["SCF"]["OUTER_SCF"]["EPS_SCF"] = "1.0E-1"

        # Setup walltime.
        input_dict["GLOBAL"]["WALLTIME"] = 86000

        builder = Cp2kBaseWorkChain.get_builder()
        builder.cp2k.code = self.inputs.cp2k_code
        builder.cp2k.structure = StructureData(ase=self.ctx.structure_with_tags)

        builder.cp2k.file = self.ctx.files
        if "settings" in self.inputs:
            builder.cp2k.settings = self.inputs.settings

        builder.cp2k.parent_calc_folder = self.ctx.ot_scf.outputs.remote_folder

        builder.cp2k.metadata.options = self.ctx.options

        # parser
        builder.cp2k.metadata.options.parser_name = "cp2k_advanced_parser"

        # cp2k input dictionary
        builder.cp2k.parameters = Dict(input_dict)

        future = self.submit(builder)
        self.to_context(diag_scf=future)

    def finalize(self):
        if not common_utils.check_if_calc_ok(self, self.ctx.diag_scf):
            self.report("diagonalization scf failed")
            return self.exit_codes.ERROR_TERMINATION  # pylint: disable=no-member

        self.out("output_parameters", self.ctx.diag_scf.outputs.output_parameters)
        self.out("remote_folder", self.ctx.diag_scf.outputs.remote_folder)
        self.out("retrieved", self.ctx.diag_scf.outputs.retrieved)
        self.report("Work chain is finished")

    # ==========================================================================


# ==========================================================================
