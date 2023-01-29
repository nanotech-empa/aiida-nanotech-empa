import copy
import os
import pathlib

import numpy as np
import yaml

from aiida.engine import ToContext, WorkChain
from aiida.orm import Bool, Code, Dict, List, SinglefileData, Str, StructureData
from aiida.plugins import CalculationFactory, WorkflowFactory

from aiida_nanotech_empa.utils import common_utils
from aiida_nanotech_empa.workflows.cp2k.cp2k_utils import (
    determine_kinds,
    dict_merge,
    get_cutoff,
    get_kinds_section,
)

Cp2kBaseWorkChain = WorkflowFactory("cp2k.base")
OverlapCalculation = CalculationFactory("spm.overlap")


class Cp2kPdosWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("cp2k_code", valid_type=Code)
        spec.input("slabsys_structure", valid_type=StructureData)
        spec.input("mol_structure", valid_type=StructureData)
        spec.input("pdos_lists", valid_type=List)
        spec.input("wfn_file_path", valid_type=Str, default=lambda: Str(""))
        spec.input("dft_params", valid_type=Dict)
        spec.input("overlap_code", valid_type=Code)
        spec.input("overlap_params", valid_type=Dict)

        spec.outline(
            cls.setup,
            cls.run_ot_scfs,
            cls.run_diags,
            cls.run_overlap,
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
        self.ctx.n_all_atoms = len(self.inputs.slabsys_structure.sites)
        self.ctx.n_mol_atoms = len(self.inputs.mol_structure.sites)

        # set up slab dft parameters
        self.ctx.slab_dft_params = copy.deepcopy(self.inputs.dft_params.get_dict())
        if "uks" not in self.ctx.slab_dft_params:
            self.ctx.slab_dft_params["uks"] = False
            self.ctx.slab_dft_params["spin_up_guess"] = []
            self.ctx.slab_dft_params["spin_dw_guess"] = []

        # set up mol dft parameters
        self.ctx.mol_dft_params = copy.deepcopy(self.inputs.dft_params.get_dict())
        
        self.ctx.mol_dft_params[
            "elpa_switch"
        ] = False  # Elpa can cause problems with small systems

        mol_spin_up = []
        mol_spin_dw = []

        if "uks" not in self.ctx.mol_dft_params:
            self.ctx.mol_dft_params["uks"] = False
            self.ctx.mol_dft_params["spin_up_guess"] = []
            self.ctx.mol_dft_params["spin_dw_guess"] = []
            slab_atoms = self.inputs.slabsys_structure.get_ase()
            mol_atoms = self.inputs.mol_structure.get_ase()

            mol_at_tuples = [
                (e, *np.round(p, 2))
                for e, p in zip(mol_atoms.get_chemical_symbols(), mol_atoms.positions)
            ]

            for i_up in self.ctx.mol_dft_params["spin_up_guess"]:
                at = slab_atoms[i_up]
                at_tup = (at.symbol, *np.round(at.position, 2))
                if at_tup in mol_at_tuples:
                    mol_spin_up.append(mol_at_tuples.index(at_tup))

            for i_dw in self.ctx.mol_dft_params["spin_dw_guess"]:
                at = slab_atoms[i_dw]
                at_tup = (at.symbol, *np.round(at.position, 2))
                if at_tup in mol_at_tuples:
                    mol_spin_dw.append(mol_at_tuples.index(at_tup))

        self.ctx.mol_dft_params["spin_up_guess"] = mol_spin_up
        self.ctx.mol_dft_params["spin_dw_guess"] = mol_spin_dw

    def run_ot_scfs(self):
        self.report("Running CP2K OT SCF")

        # whole system part
        # load input template
        with open(
            pathlib.Path(__file__).parent / "./protocols/scf_ot_protocol.yml",
            encoding="utf-8",
        ) as handle:
            protocols = yaml.safe_load(handle)
            input_dict = copy.deepcopy(protocols[self.inputs.dft_params["protocol"]])

        structure = self.inputs.slabsys_structure
        # cutoff: use the same for all calculations
        self.ctx.cutoff = get_cutoff(structure=structure)

        # get initial magnetization
        spin_up_guess = self.ctx.slab_dft_params["spin_up_guess"]
        spin_dw_guess = self.ctx.slab_dft_params["spin_dw_guess"]
        magnetization_per_site = [
            1
            if i in spin_up_guess
            else -1
            if i in spin_dw_guess
            else 0
            for i in range(self.ctx.n_all_atoms)
        ]
        structure_with_tags, kinds_dict = determine_kinds(
            structure, magnetization_per_site
        )

        ase_atoms = structure_with_tags.get_ase()
        self.ctx.slab_with_tags = ase_atoms

        builder = Cp2kBaseWorkChain.get_builder()
        builder.cp2k.code = self.inputs.cp2k_code
        builder.cp2k.structure = StructureData(ase=ase_atoms)

        builder.cp2k.file = self.ctx.files
        if self.inputs.wfn_file_path != "":
            builder.cp2k.parent_calc_folder = self.inputs.wfn_file_path.value

        input_dict["FORCE_EVAL"]["DFT"]["XC"].pop("VDW_POTENTIAL")

        # UKS
        if self.ctx.slab_dft_params["uks"]:
            input_dict["FORCE_EVAL"]["DFT"]["UKS"] = ".TRUE."
            input_dict["FORCE_EVAL"]["DFT"]["MULTIPLICITY"] = self.ctx.slab_dft_params[
                "multiplicity"
            ]

        # cutoff
        input_dict["FORCE_EVAL"]["DFT"]["MGRID"]["CUTOFF"] = self.ctx.cutoff

        # KINDS section
        self.ctx.slab_kinds_section = get_kinds_section(kinds_dict, protocol="gpw")
        dict_merge(input_dict, self.ctx.slab_kinds_section)

        # Setup walltime.
        input_dict["GLOBAL"]["WALLTIME"] = 86000

        self.ctx.slab_options = self.get_options(self.ctx.n_all_atoms)
        if self.inputs.dft_params["protocol"] == "debug":
            self.ctx.slab_options = {
            "resources": {"num_machines": 1,
            "num_mpiprocs_per_machine": 8,
            "num_cores_per_mpiproc": 1,},
            "max_wallclock_seconds": 600,
            "append_text": "cp $CP2K_DATA_DIR/BASIS_MOLOPT .",
        }
        builder.cp2k.metadata.options = self.ctx.slab_options

        # parser
        builder.cp2k.metadata.options.parser_name = "cp2k_advanced_parser"

        # cp2k input dictionary
        builder.cp2k.parameters = Dict(dict=input_dict)
        self.ctx.input_dict_slab = copy.deepcopy(input_dict)

        slab_future = self.submit(builder)
        self.to_context(slab_ot_scf=slab_future)

        # end whole system part

        # molecule part
        input_dict = copy.deepcopy(protocols[self.inputs.dft_params["protocol"]])

        structure = self.inputs.mol_structure

        # get initial magnetization
        spin_up_guess = self.ctx.mol_dft_params["spin_up_guess"]
        spin_dw_guess = self.ctx.mol_dft_params["spin_dw_guess"]
        magnetization_per_site = [
            1
            if i in spin_up_guess
            else -1
            if i in spin_dw_guess
            else 0
            for i in range(self.ctx.n_mol_atoms)
        ]
        structure_with_tags, kinds_dict = determine_kinds(
            structure, magnetization_per_site
        )

        ase_atoms = structure_with_tags.get_ase()
        self.ctx.mol_with_tags = ase_atoms

        builder = Cp2kBaseWorkChain.get_builder()
        builder.cp2k.code = self.inputs.cp2k_code
        builder.cp2k.structure = StructureData(ase=ase_atoms)

        builder.cp2k.file = self.ctx.files

        input_dict["FORCE_EVAL"]["DFT"]["XC"].pop("VDW_POTENTIAL")

        # UKS
        if self.ctx.mol_dft_params["uks"]:
            input_dict["FORCE_EVAL"]["DFT"]["UKS"] = ".TRUE."
            input_dict["FORCE_EVAL"]["DFT"]["MULTIPLICITY"] = self.ctx.slab_dft_params[
                "multiplicity"
            ]

        # cutoff
        input_dict["FORCE_EVAL"]["DFT"]["MGRID"]["CUTOFF"] = self.ctx.cutoff

        # KINDS section
        self.ctx.mol_kinds_section = get_kinds_section(kinds_dict, protocol="gpw")
        dict_merge(input_dict, self.ctx.mol_kinds_section)

        # Setup walltime.
        input_dict["GLOBAL"]["WALLTIME"] = 86000
        self.ctx.mol_options = self.get_options(self.ctx.n_mol_atoms)
        if self.inputs.dft_params["protocol"] == "debug":
            self.ctx.mol_options = {
            "resources": {"num_machines": 1,
            "num_mpiprocs_per_machine": 1,
            "num_cores_per_mpiproc": 1,},
            "max_wallclock_seconds": 600,
            "append_text": "cp $CP2K_DATA_DIR/BASIS_MOLOPT .",
        }        
        builder.cp2k.metadata.options = self.ctx.mol_options

        # parser
        builder.cp2k.metadata.options.parser_name = "cp2k_advanced_parser"

        # cp2k input dictionary
        builder.cp2k.parameters = Dict(dict=input_dict)
        self.ctx.input_dict_mol = copy.deepcopy(input_dict)

        mol_future = self.submit(builder)
        self.to_context(mol_ot_scf=mol_future)
        # end molecule part

    def run_diags(self):
        for calculation in [self.ctx.slab_ot_scf, self.ctx.mol_ot_scf]:
            if not common_utils.check_if_calc_ok(self, calculation):
                return self.exit_codes.ERROR_TERMINATION  # pylint: disable=no-member
        emax = float(self.inputs.overlap_params.get_dict()["--emax1"])
        nlumo = int(self.inputs.overlap_params.get_dict()["--nlumo2"])
        added_mos = np.max([100, int(1.2 * self.ctx.n_all_atoms * emax / 5.0)])

        self.report("Running CP2K diagonalization SCF")

        # whole system part
        # load input template
        with open(
            pathlib.Path(__file__).parent / "./protocols/scf_diag_protocol.yml",
            encoding="utf-8",
        ) as handle:
            protocols = yaml.safe_load(handle)
            scf_dict = copy.deepcopy(protocols[self.inputs.dft_params["protocol"]])

        input_dict = copy.deepcopy(self.ctx.input_dict_slab)
        if self.ctx.slab_dft_params["elpa_switch"]:
            input_dict["GLOBAL"]["PREFERRED_DIAG_LIBRARY"] = "ELPA"
            input_dict["GLOBAL"]["ELPA_KERNEL"] = "AUTO"
            input_dict["GLOBAL"]["DBCSR"] = {"USE_MPI_ALLOCATOR": ".FALSE."}
        input_dict["FORCE_EVAL"]["DFT"].pop("SCF")
        input_dict["FORCE_EVAL"]["DFT"]["SCF"] = scf_dict
        input_dict["FORCE_EVAL"]["DFT"]["SCF"]["ADDED_MOS"] = added_mos
        # pdos
        if self.inputs.pdos_lists is not None:
            pdos_list_dicts = [
                {"COMPONENTS": "", "LIST": e} for e in self.inputs.pdos_lists
            ]
            input_dict["FORCE_EVAL"]["DFT"]["PRINT"]["PDOS"] = {
                "NLUMO": added_mos,
                "LDOS": pdos_list_dicts,
            }

        smearing = "smear_t" in self.ctx.slab_dft_params
        if smearing:
            input_dict["FORCE_EVAL"]["DFT"]["SCF"]["SMEAR"][
                "ELECTRONIC_TEMPERATURE"
            ] = self.ctx.slab_dft_params["smear_t"]
        else:
            input_dict["FORCE_EVAL"]["DFT"]["SCF"].pop("SMEAR")

        # UKS
        if self.ctx.slab_dft_params["uks"] and smearing and self.ctx.slab_dft_params["force_multiplicity"]:
            input_dict["FORCE_EVAL"]["DFT"]["SCF"]["SMEAR"][
                "FIXED_MAGNETIC_MOMENT"
            ] = (self.ctx.slab_dft_params["multiplicity"] - 1)
        # no self consistent diag
        if not self.ctx.slab_dft_params['sc_diag']:
            input_dict["FORCE_EVAL"]["DFT"]["SCF"].pop("SMEAR")
            input_dict["FORCE_EVAL"]["DFT"]["SCF"]["EPS_SCF"] = "1.0E-1"
            input_dict["FORCE_EVAL"]["DFT"]["SCF"]["OUTER_SCF"]["EPS_SCF"] = "1.0E-1"

        if not smearing and "SMEAR" in input_dict["FORCE_EVAL"]["DFT"]["SCF"]:
            input_dict["FORCE_EVAL"]["DFT"]["SCF"].pop("SMEAR")

        builder = Cp2kBaseWorkChain.get_builder()
        builder.cp2k.code = self.inputs.cp2k_code
        builder.cp2k.structure = StructureData(ase=self.ctx.slab_with_tags)

        builder.cp2k.file = self.ctx.files
        builder.cp2k.settings = Dict(dict={"additional_retrieve_list": ["*.pdos"]})
        builder.cp2k.parent_calc_folder = self.ctx.slab_ot_scf.outputs.remote_folder

        builder.cp2k.metadata.options = self.ctx.slab_options

        # parser
        builder.cp2k.metadata.options.parser_name = "cp2k_advanced_parser"

        # cp2k input dictionary
        builder.cp2k.parameters = Dict(dict=input_dict)

        slab_future = self.submit(builder)
        self.to_context(slab_diag_scf=slab_future)

        # end whole system part

        # molecule part
        scf_dict = copy.deepcopy(protocols[self.inputs.dft_params["protocol"]])
        input_dict = copy.deepcopy(self.ctx.input_dict_mol)
        # no ELPA for the molecule
        input_dict["FORCE_EVAL"]["DFT"].pop("SCF")
        input_dict["FORCE_EVAL"]["DFT"]["SCF"] = scf_dict
        input_dict["FORCE_EVAL"]["DFT"]["SCF"]["ADDED_MOS"] = nlumo + 2

        smearing = "smear_t" in self.ctx.mol_dft_params
        if smearing:
            input_dict["FORCE_EVAL"]["DFT"]["SCF"]["SMEAR"][
                "ELECTRONIC_TEMPERATURE"
            ] = self.ctx.mol_dft_params["smear_t"]
        else:
            input_dict["FORCE_EVAL"]["DFT"]["SCF"].pop("SMEAR")

        # UKS
        if self.ctx.mol_dft_params["uks"] and smearing and self.ctx.slab_dft_params["force_multiplicity"]:
            input_dict["FORCE_EVAL"]["DFT"]["SCF"]["SMEAR"][
                "FIXED_MAGNETIC_MOMENT"
            ] = (self.ctx.mol_dft_params["multiplicity"] - 1)
        # no self consistent diag
        if not self.ctx.slab_dft_params['sc_diag']:
            input_dict["FORCE_EVAL"]["DFT"]["SCF"].pop("SMEAR")
            input_dict["FORCE_EVAL"]["DFT"]["SCF"]["EPS_SCF"] = "1.0E-1"
            input_dict["FORCE_EVAL"]["DFT"]["SCF"]["OUTER_SCF"]["EPS_SCF"] = "1.0E-1"

        if not smearing and "SMEAR" in input_dict["FORCE_EVAL"]["DFT"]["SCF"]:
            input_dict["FORCE_EVAL"]["DFT"]["SCF"].pop("SMEAR")

        builder = Cp2kBaseWorkChain.get_builder()
        builder.cp2k.code = self.inputs.cp2k_code
        builder.cp2k.structure = StructureData(ase=self.ctx.mol_with_tags)

        builder.cp2k.file = self.ctx.files
        builder.cp2k.parent_calc_folder = self.ctx.mol_ot_scf.outputs.remote_folder

        builder.cp2k.metadata.options = self.ctx.mol_options

        # parser
        builder.cp2k.metadata.options.parser_name = "cp2k_advanced_parser"

        # cp2k input dictionary
        builder.cp2k.parameters = Dict(dict=input_dict)

        mol_future = self.submit(builder)
        self.to_context(mol_diag_scf=mol_future)
        # end molecule part

    def run_overlap(self):

        for calculation in [self.ctx.slab_diag_scf, self.ctx.mol_diag_scf]:
            if not common_utils.check_if_calc_ok(self, calculation):
                return self.exit_codes.ERROR_TERMINATION  # pylint: disable=no-member

        self.report("Running overlap")

        inputs = {}
        inputs["metadata"] = {}
        inputs["metadata"]["label"] = "overlap"
        inputs["code"] = self.inputs.overlap_code
        inputs["parameters"] = self.inputs.overlap_params
        inputs["parent_slab_folder"] = self.ctx.slab_diag_scf.outputs.remote_folder
        inputs["parent_mol_folder"] = self.ctx.mol_diag_scf.outputs.remote_folder

        n_machines = 4 if self.ctx.n_all_atoms < 2000 else 8

        inputs["metadata"]["options"] = {
            "resources": {"num_machines": n_machines},
            "max_wallclock_seconds": 86400,
        }

        if self.inputs.dft_params["protocol"] == "debug":
            inputs["metadata"]["options"]["max_wallclock_seconds"] = 600
        settings = Dict(dict={"additional_retrieve_list": ["overlap.npz"]})
        inputs["settings"] = settings

        #self.report("overlap inputs: " + str(inputs))

        future = self.submit(OverlapCalculation, **inputs)
        return ToContext(overlap=future)

    def finalize(self):
        if "overlap.npz" not in [obj.name for obj in self.ctx.overlap.outputs.retrieved.list_objects()]:
            self.report("Overlap calculation did not finish correctly")
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
