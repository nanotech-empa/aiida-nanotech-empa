import pathlib

import numpy as np
from aiida import engine, orm, plugins

from ...utils import common_utils
from . import cp2k_utils

Cp2kBaseWorkChain = plugins.WorkflowFactory("cp2k.base")


class Cp2kGeoOptWorkChain(engine.WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("code", valid_type=orm.Code)
        spec.input("structure", valid_type=orm.StructureData)
        spec.input("parent_calc_folder", valid_type=orm.RemoteData, required=False)
        spec.input(
            "protocol",
            valid_type=orm.Str,
            default=lambda: orm.Str("standard"),
            required=False,
            help="Protocol supported by the work chain (geo_opt_protocol).",
        )
        spec.input("dft_params", valid_type=orm.Dict)
        spec.input("sys_params", valid_type=orm.Dict)
        spec.input(
            "options",
            valid_type=dict,
            non_db=True,
            help="Define options for the cacluations: walltime, memory, CPUs, etc.",
        )

        # Workchain outline.
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
            "basis": orm.SinglefileData(
                file=pathlib.Path(__file__).parent / "data" / "BASIS_MOLOPT"
            ),
            "pseudo": orm.SinglefileData(
                file=pathlib.Path(__file__).parent / "data" / "POTENTIAL"
            ),
        }

        self.ctx.sys_params = self.inputs.sys_params.get_dict()
        self.ctx.dft_params = self.inputs.dft_params.get_dict()
        self.ctx.input_dict = cp2k_utils.load_protocol(
            "geo_opt_protocol.yml", self.inputs.protocol.value
        )

        # vdW section.
        if "vdw" in self.ctx.dft_params:
            if not self.ctx.dft_params["vdw"]:
                self.ctx.input_dict["FORCE_EVAL"]["DFT"]["XC"].pop("VDW_POTENTIAL")
        else:
            self.ctx.input_dict["FORCE_EVAL"]["DFT"]["XC"].pop("VDW_POTENTIAL")

        # Charge.
        if "charge" in self.ctx.dft_params:
            self.ctx.input_dict["FORCE_EVAL"]["DFT"]["CHARGE"] = self.ctx.dft_params[
                "charge"
            ]

        # UKS.
        magnetization_per_site = [0 for i in range(len(self.inputs.structure.sites))]
        if "uks" in self.ctx.dft_params and self.ctx.dft_params["uks"]:
            magnetization_per_site = self.ctx.dft_params["magnetization_per_site"]
            self.ctx.input_dict["FORCE_EVAL"]["DFT"]["UKS"] = ".TRUE."
            self.ctx.input_dict["FORCE_EVAL"]["DFT"][
                "MULTIPLICITY"
            ] = self.ctx.dft_params["multiplicity"]

        # Get initial magnetization.
        structure_with_tags, kinds_dict = cp2k_utils.determine_kinds(
            self.inputs.structure, magnetization_per_site
        )

        ase_atoms = structure_with_tags.get_ase()

        # Non-periodic systems only NONE and XYZ implemented:
        if "periodic" in self.ctx.dft_params:
            if self.ctx.dft_params["periodic"] == "NONE":
                # Make sure cell is big enough for MT poisson solver and center molecule.
                if self.inputs.protocol.value == "debug":
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
        self.ctx.kinds_section = cp2k_utils.get_kinds_section(
            kinds_dict, protocol="gpw"
        )
        cp2k_utils.dict_merge(self.ctx.input_dict, self.ctx.kinds_section)

        # Overwrite cutoff if given in dft_params.
        cutoff = cp2k_utils.get_cutoff(structure=self.inputs.structure)
        if "cutoff" in self.ctx.dft_params:
            cutoff = self.ctx.dft_params["cutoff"]

        self.ctx.input_dict["FORCE_EVAL"]["DFT"]["MGRID"]["CUTOFF"] = cutoff

        # Cell symmetry.
        if "symmetry" in self.ctx.sys_params:
            self.ctx.input_dict["FORCE_EVAL"]["SUBSYS"]["CELL"][
                "SYMMETRY"
            ] = self.ctx.sys_params["symmetry"]

        # Cell optimization.
        if "cell_opt" in self.ctx.sys_params:
            cell_input_dict = cp2k_utils.load_protocol(
                "cell_opt_protocol.yml", self.inputs.protocol.value
            )
            self.ctx.input_dict["GLOBAL"]["RUN_TYPE"] = "CELL_OPT"
            self.ctx.input_dict["MOTION"] = cell_input_dict["MOTION"]
            self.ctx.input_dict["FORCE_EVAL"]["STRESS_TENSOR"] = "ANALYTICAL"
            if "cell_opt_constraint" in self.ctx.sys_params:
                self.ctx.input_dict["MOTION"]["CELL_OPT"][
                    "CONSTRAINT"
                ] = self.ctx.sys_params["cell_opt_constraint"]
            for sym in ["KEEP_SYMMETRY", "KEEP_ANGLES", "KEEP_SPACE_GROUP"]:
                if sym in self.ctx.sys_params:
                    self.ctx.input_dict["MOTION"]["CELL_OPT"][sym] = ""

        # Constraints.
        if "constraints" in self.ctx.sys_params:
            self.ctx.input_dict["MOTION"][
                "CONSTRAINT"
            ] = cp2k_utils.get_constraints_section(self.ctx.sys_params["constraints"])
        # Colvars.
        if "colvars" in self.ctx.sys_params:
            self.ctx.input_dict["FORCE_EVAL"]["SUBSYS"].update(
                cp2k_utils.get_colvars_section(self.ctx.sys_params["colvars"])
            )

        # Resources.
        self.ctx.options = self.inputs.options
        self.ctx.input_dict["GLOBAL"]["WALLTIME"] = max(
            600, self.ctx.options["max_wallclock_seconds"] - 600
        )

    def submit_calc(self):
        self.report("Submitting geometry optimization")

        builder = Cp2kBaseWorkChain.get_builder()
        builder.cp2k.code = self.inputs.code
        builder.cp2k.structure = orm.StructureData(ase=self.ctx.structure_with_tags)
        builder.cp2k.file = self.ctx.files

        # Resources.
        builder.cp2k.metadata.options = self.ctx.options

        # Parser.
        builder.cp2k.metadata.options.parser_name = "cp2k_advanced_parser"

        # Handlers.
        builder.handler_overrides = orm.Dict({"restart_incomplete_calculation": True})

        # Restart wfn.
        if "parent_calc_folder" in self.inputs:
            builder.cp2k.parent_calc_folder = self.inputs.parent_calc_folder

        # CP2K input dictionary.
        builder.cp2k.parameters = orm.Dict(self.ctx.input_dict)

        future = self.submit(builder)
        self.to_context(geo_opt=future)

    def finalize(self):
        self.report("Finalizing.")

        if not self.ctx.geo_opt.is_finished_ok:
            return self.exit_codes.ERROR_TERMINATION

        for out in self.ctx.geo_opt.outputs:
            self.out(out, self.ctx.geo_opt.outputs[out])

        # Add extras.
        struc = self.inputs.structure
        common_utils.add_extras(struc, "surfaces", self.node.uuid)

        return engine.ExitCode(0)
