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

Cp2kDiagWorkChain = WorkflowFactory("nanotech_empa.cp2k.diag")
OverlapCalculation = CalculationFactory("spm.overlap")


class Cp2kPdosWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("cp2k_code", valid_type=Code)
        spec.input("slabsys_structure", valid_type=StructureData)
        spec.input("mol_structure", valid_type=StructureData)
        spec.input("pdos_lists", valid_type=List)
        spec.input("wfn_file_path", valid_type=Str, required=False)
        spec.input("dft_params", valid_type=Dict)
        spec.input("overlap_code", valid_type=Code)
        spec.input("overlap_params", valid_type=Dict)
        spec.input_namespace(
            "options",
            valid_type=dict,
            non_db=True,
            help=
            "Define options for the cacluations: walltime, memory, CPUs, etc.")        

        spec.outline(
            cls.setup,
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
        
        self.ctx.n_slab_atoms = len(self.inputs.slabsys_structure.sites)
        self.ctx.slab_options = self.inputs.options['slab']
        self.ctx.mol_options = self.inputs.options['molecule']
        emax = float(self.inputs.overlap_params.get_dict()["--emax1"])
        nlumo = int(self.inputs.overlap_params.get_dict()["--nlumo2"])
        

        # set up slab dft parameters
        self.ctx.slab_dft_params = copy.deepcopy(self.inputs.dft_params.get_dict())
        self.ctx.slab_dft_params["added_mos"] =  np.max([100, int(1.2 * self.ctx.n_slab_atoms * emax / 5.0)])

        # set up mol dft parameters
        self.ctx.mol_dft_params = copy.deepcopy(self.ctx.slab_dft_params)
        self.ctx.mol_dft_params["added_mos"] = nlumo + 2
        # force same cutoff for molecule and slab
        self.ctx.mol_dft_params["cutoff"] = get_cutoff(self.inputs.slabsys_structure)        
        self.ctx.mol_dft_params[
            "elpa_switch"
        ] = False  # Elpa can cause problems with small systems




        slab_atoms = self.inputs.slabsys_structure.get_ase()
        mol_atoms = self.inputs.mol_structure.get_ase()

        if   self.ctx.mol_dft_params["uks"]:
            mol_spin_up = []
            mol_spin_dw = []
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

    def run_diags(self):
        # slab part
        self.report("Running Diag Workchain for slab")

        builder = Cp2kDiagWorkChain.get_builder()
        builder.cp2k_code = self.inputs.cp2k_code
        builder.structure = self.inputs.slabsys_structure
        builder.dft_params = Dict(dict=self.ctx.slab_dft_params)
        builder.settings = Dict(dict={"additional_retrieve_list": ["*.pdos"]})
        builder.options = Dict(dict=self.ctx.slab_options)        
        # pdos
        if self.inputs.pdos_lists is not None:
            builder.pdos_lists = List(list=[pdos[0] for pdos in self.inputs.pdos_lists])

        slab_future = self.submit(builder)
        self.to_context(slab_diag_scf=slab_future)
        # end slab part

        # molecule part
        self.report("Running Diag Workchain for molecule")

        builder = Cp2kDiagWorkChain.get_builder()
        builder.cp2k_code = self.inputs.cp2k_code
        builder.structure = self.inputs.mol_structure
        builder.dft_params = Dict(dict=self.ctx.mol_dft_params)
        builder.options = Dict(dict=self.ctx.mol_options)       
        
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

        n_machines = 4 if self.ctx.n_slab_atoms < 2000 else 8

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
        # Add the workchain pk to the input structure extras
        extras_label = "Cp2kPdosWorkChain_uuids"
        if extras_label not in self.inputs.slabsys_structure.extras:
            extras_list = []
        else:
            extras_list = self.inputs.slabsys_structure.extras[extras_label]
        extras_list.append(self.node.uuid)
        self.inputs.slabsys_structure.set_extra(extras_label, extras_list)            
        self.report("Work chain is finished")
