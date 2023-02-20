import copy
import os
import pathlib

import numpy as np
import yaml

from aiida.engine import ToContext, WorkChain, while_
from aiida.orm import Bool, Code, Dict, List, SinglefileData, Str, StructureData
from aiida.orm.nodes.data.array import ArrayData
from aiida.orm import SinglefileData
from aiida.orm import RemoteData

from aiida.plugins import CalculationFactory, WorkflowFactory

from aiida_nanotech_empa.utils import common_utils
from aiida_nanotech_empa.workflows.cp2k.cp2k_utils import (
    determine_kinds,
    dict_merge,
    get_cutoff,
    get_kinds_section,
)

Cp2kDiagWorkChain = WorkflowFactory("nanotech_empa.cp2k.diag")
StmCalculation = CalculationFactory("nanotech_empa.stm")


class Cp2kOrbitalsWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super(Cp2kOrbitalsWorkChain, cls).define(spec)

        spec.input("cp2k_code", valid_type=Code)
        spec.input("structure", valid_type=StructureData)
        spec.input("wfn_file_path", valid_type=Str, required=False)
        spec.input("dft_params", valid_type=Dict)
        spec.input("spm_code", valid_type=Code)
        spec.input("spm_params", valid_type=Dict)
        spec.input(
            "options",
            valid_type=dict,
            non_db=True,
            help="Define options for the cacluations: walltime, memory, CPUs, etc.",
        )

        spec.outline(
            cls.setup,
            cls.run_diag_scf,
            cls.run_stm,
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
        structure = self.inputs.structure
        n_lumo = int(self.inputs.spm_params.get_dict()["--n_lumo"])
        added_mos = np.max([n_lumo, 20])
        self.ctx.dft_params = self.inputs.dft_params.get_dict()
        self.ctx.dft_params["added_mos"] = added_mos

    def run_diag_scf(self):
        self.report("Running CP2K diagonalization SCF")
        builder = Cp2kDiagWorkChain.get_builder()
        builder.cp2k_code = self.inputs.cp2k_code
        builder.structure = self.inputs.structure
        builder.dft_params = Dict(self.ctx.dft_params)
        builder.settings = Dict(
            {
                "additional_retrieve_list": [
                    "aiida.inp",
                    "BASIS_MOLOPT",
                    "aiida.coords.xyz",
                    "aiida-RESTART.wfn",
                ]
            }
        )
        builder.options = Dict(self.inputs.options)

        future = self.submit(builder)
        self.to_context(diag_scf=future)

    def run_stm(self):
        self.report("STM calculation")
        if not common_utils.check_if_calc_ok(self, self.ctx.diag_scf):
            return self.exit_codes.ERROR_TERMINATION  # pylint: disable=no-member

        inputs = {}
        inputs["metadata"] = {}
        inputs["metadata"]["label"] = "orb"
        inputs["code"] = self.inputs.spm_code
        inputs["parameters"] = self.inputs.spm_params
        inputs["parent_calc_folder"] = self.ctx.diag_scf.outputs.remote_folder
        inputs["metadata"]["options"] = {
            "resources": {"num_machines": 1},
            "max_wallclock_seconds": 3600,
        }

        # Need to make an explicit instance for the node to be stored to aiida
        settings = Dict({"additional_retrieve_list": ["orb.npz"]})
        inputs["settings"] = settings

        future = self.submit(StmCalculation, **inputs)
        return ToContext(stm=future)

    def finalize(self):
        if "orb.npz" not in [
            obj.name for obj in self.ctx.stm.outputs.retrieved.list_objects()
        ]:
            self.report("Orbital calculation did not finish correctly")
            return self.exit_codes.ERROR_TERMINATION
        self.out("dft_output_parameters", self.ctx.diag_scf.outputs.output_parameters)
        self.out("retrieved", self.ctx.diag_scf.outputs.retrieved)
        # Add the workchain pk to the input structure extras
        extras_label = "Cp2kOrbitalsWorkChain_uuids"
        if extras_label not in self.inputs.structure.extras:
            extras_list = []
        else:
            extras_list = self.inputs.structure.extras[extras_label]
        extras_list.append(self.node.uuid)
        self.inputs.structure.set_extra(extras_label, extras_list)
        self.report("Work chain is finished")

    # ==========================================================================
