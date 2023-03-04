import os
import pathlib
import yaml
import copy
import numpy as np

from aiida.engine import WorkChain, ToContext, ExitCode
from aiida.orm import Int, Bool, Code, Dict, List, Str
from aiida.orm import SinglefileData, StructureData
from aiida.plugins import WorkflowFactory
from aiida_nanotech_empa.workflows.cp2k.cp2k_utils import (
    get_kinds_section,
    determine_kinds,
    dict_merge,
    get_cutoff,
)
from aiida_nanotech_empa.workflows.cp2k.cp2k_utils import (
    get_colvars_section,
    get_constraints_section,
)

from aiida_nanotech_empa.utils import common_utils

Cp2kBaseWorkChain = WorkflowFactory("cp2k.base")


class Cp2kGeoOptWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("code", valid_type=Code)
        spec.input("structure", valid_type=StructureData)
        spec.input("wfn_file_path", valid_type=Str, required=False)
        spec.input("dft_params", valid_type=Dict)
        spec.input("sys_params", valid_type=Dict)
        spec.input(
            "options",
            valid_type=dict,
            non_db=True,
            help="Define options for the cacluations: walltime, memory, CPUs, etc.",
        )

        # workchain outline
        spec.outline(cls.setup, cls.submit_calc, cls.finalize)
        spec.outputs.dynamic = True

        spec.exit_code(
            390,
            "ERROR_TERMINATION",
            message="One or more steps of the work chain failed.",
        )

    def setup(self):
        self.report("Inspecting input and setting up things")

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

        self.ctx.sys_params = self.inputs.sys_params.get_dict()
        self.ctx.dft_params = self.inputs.dft_params.get_dict()

        self.ctx.n_atoms = len(self.inputs.structure.sites)

        # load input template
        with open(
            pathlib.Path(__file__).parent / "./protocols/geo_opt_protocol.yml",
            encoding="utf-8",
        ) as handle:
            protocols = yaml.safe_load(handle)
            self.ctx.input_dict = copy.deepcopy(
                protocols[self.ctx.dft_params["protocol"]]
            )

        # vdW section
        if "vdw" in self.ctx.dft_params:
            if not self.ctx.dft_params["vdw"]:
                self.ctx.input_dict["FORCE_EVAL"]["DFT"]["XC"].pop("VDW_POTENTIAL")
        else:
            self.ctx.input_dict["FORCE_EVAL"]["DFT"]["XC"].pop("VDW_POTENTIAL")

        # charge
        if "charge" in self.ctx.dft_params:
            self.ctx.input_dict["FORCE_EVAL"]["DFT"]["CHARGE"] = self.ctx.dft_params[
                "charge"
            ]

        # uks
        magnetization_per_site = [0 for i in range(self.ctx.n_atoms)]
        if "uks" in self.ctx.dft_params:
            if self.ctx.dft_params["uks"]:
                magnetization_per_site = self.ctx.dft_params["magnetization_per_site"]
                self.ctx.input_dict["FORCE_EVAL"]["DFT"]["UKS"] = ".TRUE."
                self.ctx.input_dict["FORCE_EVAL"]["DFT"][
                    "MULTIPLICITY"
                ] = self.ctx.dft_params["multiplicity"]

        # get initial magnetization
        structure_with_tags, kinds_dict = determine_kinds(
            self.inputs.structure, magnetization_per_site
        )

        ase_atoms = structure_with_tags.get_ase()

        # non periodic systems only NONE and XYZ implemented:
        if "periodic" in self.ctx.dft_params:
            if self.ctx.dft_params["periodic"] == "NONE":
                # make sure cell is big enough for MT poisson solver and center molecule
                if self.ctx.dft_params["protocol"] == "debug":
                    extra_cell = 5.0
                else:
                    extra_cell = 15.0
                ase_atoms.cell = 2 * (np.ptp(ase_atoms.positions, axis=0)) + extra_cell
                ase_atoms.center()

                # Poisson solver
                self.ctx.input_dict["FORCE_EVAL"]["SUBSYS"]["CELL"]["PERIODIC"] = "NONE"
                self.ctx.input_dict["FORCE_EVAL"]["DFT"]["POISSON"]["PERIODIC"] = "NONE"
                self.ctx.input_dict["FORCE_EVAL"]["DFT"]["POISSON"][
                    "POISSON_SOLVER"
                ] = "MT"
            # to be done: more cases

        self.ctx.structure_with_tags = ase_atoms
        self.ctx.kinds_section = get_kinds_section(kinds_dict, protocol="gpw")
        dict_merge(self.ctx.input_dict, self.ctx.kinds_section)

        # get cutoff
        cutoff = get_cutoff(structure=self.inputs.structure)

        # overwrite cutoff if given in dft_params
        if "cutoff" in self.ctx.dft_params:
            cutoff = self.ctx.dft_params["cutoff"]

        self.ctx.input_dict["FORCE_EVAL"]["DFT"]["MGRID"]["CUTOFF"] = cutoff

        # cell symmetry:
        if "symmetry" in self.ctx.sys_params:
            self.ctx.input_dict["FORCE_EVAL"]["SUBSYS"]["CELL"][
                "SYMMETRY"
            ] = self.ctx.sys_params["symmetry"]

        # cell optimization:
        if "cell_opt" in self.ctx.sys_params:
            with open(
                pathlib.Path(__file__).parent / "./protocols/cell_opt_protocol.yml",
                encoding="utf-8",
            ) as handle:
                protocols = yaml.safe_load(handle)
                cell_input_dict = copy.deepcopy(
                    protocols[self.ctx.dft_params["protocol"]]
                )
            self.ctx.input_dict["MOTION"] = cell_input_dict["MOTION"]
            self.ctx.input_dict["FORCE_EVAL"]["STRESS_TENSOR"] = "ANALYTICAL"
            if "cell_opt_constraint" in self.ctx.sys_params:
                self.ctx.input_dict["MOTION"]["CELL_OPT"][
                    "CONSTRAINT"
                ] = self.ctx.sys_params["cell_opt_constraint"]
            for sym in ["KEEP_SYMMETRY", "KEEP_ANGLES", "KEEP_SPACE_GROUP"]:
                if sym in self.ctx.sys_params:
                    self.ctx.input_dict["MOTION"]["CELL_OPT"][sym] = ""

        # constraints
        if "constraints" in self.ctx.sys_params:
            self.ctx.input_dict["MOTION"]["CONSTRAINT"] = get_constraints_section(
                self.ctx.sys_params["constraints"]
            )
        # colvars
        if "colvars" in self.ctx.sys_params:
            self.ctx.input_dict["FORCE_EVAL"]["SUBSYS"].update(
                get_colvars_section(self.ctx.sys_params["colvars"])
            )

        # resources
        self.ctx.options = self.inputs.options
        if self.ctx.dft_params["protocol"] == "debug":
            self.ctx.options = {
                "max_wallclock_seconds": 600,
                "resources": {
                    "num_machines": 1,
                    "num_mpiprocs_per_machine": 1,
                    "num_cores_per_mpiproc": 1,
                },
            }
        self.ctx.input_dict["GLOBAL"]["WALLTIME"] = self.ctx.options[
            "max_wallclock_seconds"
        ]
        # --------------------------------------------------

    def submit_calc(self):
        self.report("Submitting geometry optimization")

        builder = Cp2kBaseWorkChain.get_builder()
        builder.cp2k.code = self.inputs.code
        builder.cp2k.structure = StructureData(ase=self.ctx.structure_with_tags)
        builder.cp2k.file = self.ctx.files

        # resources
        builder.cp2k.metadata.options = self.ctx.options

        # parser
        builder.cp2k.metadata.options.parser_name = "cp2k_advanced_parser"

        # handlers
        builder.handler_overrides = Dict({"restart_incomplete_calculation": True})

        # restart wfn
        if "wfn_file_path" in self.inputs:
            builder.cp2k.parent_calc_folder = self.inputs.wfn_file_path.value

        # cp2k input dictionary
        builder.cp2k.parameters = Dict(self.ctx.input_dict)

        future = self.submit(builder)
        self.to_context(geo_opt=future)

    def finalize(self):
        self.report("Finalizing.")

        if not self.ctx.geo_opt.is_finished_ok:
            return self.exit_codes.ERROR_TERMINATION

        for out in self.ctx.geo_opt.outputs:
            self.out(out, self.ctx.geo_opt.outputs[out])

        # Add extras
        struc = self.inputs.structure
        #ase_geom = struc.get_ase()
        #struc.set_extra("thumbnail", common_utils.thumbnail(ase_struc=ase_geom))
        common_utils.add_extras(struc, "surfaces", self.node.uuid)

        return ExitCode(0)
