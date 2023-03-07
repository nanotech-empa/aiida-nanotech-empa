import numpy as np
from copy import deepcopy

from aiida.engine import WorkChain, ExitCode, if_
from aiida.orm import Code, Dict, List, Str
from aiida.orm import StructureData, load_node
from aiida.plugins import CalculationFactory
from aiida_nanotech_empa.workflows.cp2k.cp2k_utils import (
    get_colvars_section,
    get_constraints_section,
)
from aiida_nanotech_empa.workflows.cp2k.cp2k_utils import (
    make_geom_file,
    mk_wfn_cp_commands,
    get_dft_inputs,
)

from aiida_nanotech_empa.utils import common_utils

Cp2kCalculation = CalculationFactory("cp2k")


class Cp2kNebWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("code", valid_type=Code)
        spec.input("structure", valid_type=StructureData)
        spec.input("replica_uuids", valid_type=List)
        spec.input("wfn_cp_commands", valid_type=Str, required=False)
        spec.input("restart_from", valid_type=Str, required=False)
        spec.input("dft_params", valid_type=Dict)
        spec.input("sys_params", valid_type=Dict)
        spec.input("neb_params", valid_type=Dict)
        spec.input(
            "options",
            valid_type=dict,
            non_db=True,
            help="Define options for the cacluations: walltime, memory, CPUs, etc.",
        )

        # workchain outline
        spec.outline(
            cls.setup,
            if_(cls.should_run_scf)(cls.first_scf),
            cls.submit_neb,
            cls.finalize,
        )
        spec.outputs.dynamic = True

        spec.exit_code(
            390,
            "ERROR_TERMINATION",
            message="One or more steps of the work chain failed.",
        )
        spec.exit_code(
            380,
            "ERROR_UUIDS",
            message="no structures specified",
        )

    def setup(self):
        self.report("Inspecting input and setting up things")
        if "restart_from" in self.inputs:
            dft_params = load_node(
                self.inputs.restart_from.value
            ).inputs.dft_params.get_dict()
        else:
            dft_params = self.inputs.dft_params.get_dict()

        (
            self.ctx.files,
            self.ctx.input_dict,
            self.ctx.structure_with_tags,
        ) = get_dft_inputs(dft_params, self.inputs.structure, "neb_protocol.yml")
        self.ctx.sys_params = self.inputs.sys_params.get_dict()
        self.ctx.neb_params = self.inputs.neb_params.get_dict()

        uuids_to_check = [self.inputs.structure.uuid]
        # check if restarting
        if len(self.inputs.replica_uuids) > 0:
            uuids_to_check += self.inputs.replica_uuids
        elif "restart_from" in self.inputs:
            self.report("checking wfn available from previos neb")
            workcalc = load_node(self.inputs.restart_from.value)
            n_previous_replica = workcalc.inputs.neb_params["number_of_replica"]
            uuids_to_check = [
                workcalc.outputs["opt_replica_%s" % str(i).zfill(3)].uuid
                for i in range(1, n_previous_replica)
            ]
        else:
            return self.exit_codes.ERROR_UUIDS

        # check for existing wfn files and create copy commands
        self.ctx.should_run_scf = False
        self.ctx.wfn_cp_commands = mk_wfn_cp_commands(
            self.ctx.neb_params["number_of_replica"],
            uuids_to_check,
            self.inputs.code.computer,
        )
        if len(self.ctx.wfn_cp_commands) == 0:
            self.ctx.should_run_scf = True

        # replica files with tags must be after structure_with_tags.
        tags = self.ctx.structure_with_tags.get_tags().astype(np.int32).tolist()
        self.ctx.files["replica_001_xyz"] = make_geom_file(
            self.ctx.structure_with_tags, "replica_001.xyz", tags=tags
        )
        self.ctx.input_dict["MOTION"]["BAND"]["REPLICA"] = [
            {"COORD_FILE_NAME": "replica_001.xyz"}
        ]
        for i, uuid in enumerate(self.inputs.replica_uuids):
            structure = load_node(uuid).get_ase()
            filename = "replica_%s.xyz" % str(i + 2).zfill(3)
            self.ctx.files[filename.replace(".", "_")] = make_geom_file(
                structure, filename, tags=tags
            )
            # and update input dictionary.
            self.ctx.input_dict["MOTION"]["BAND"]["REPLICA"].append(
                {"COORD_FILE_NAME": filename}
            )

        # constraints.
        if "constraints" in self.ctx.sys_params:
            self.ctx.input_dict["MOTION"]["CONSTRAINT"] = get_constraints_section(
                self.ctx.sys_params["constraints"]
            )
        # colvars.
        if "colvars" in self.ctx.sys_params:
            self.ctx.input_dict["FORCE_EVAL"]["SUBSYS"].update(
                get_colvars_section(self.ctx.sys_params["colvars"])
            )

        # neb parameters.
        for param in [
            "align_frames",
            "band_type",
            "k_spring",
            "nproc_rep",
            "number_of_replica",
        ]:
            if param in self.ctx.neb_params:
                self.ctx.input_dict["MOTION"]["BAND"][
                    param.upper()
                ] = self.ctx.neb_params[param]

        if "nsteps_it" in self.ctx.neb_params:
            self.ctx.input_dict["MOTION"]["BAND"]["CI_NEB"] = {
                "NSTEPS_IT": self.ctx.neb_params["nsteps_it"]
            }
        if "optimize_end_points" in self.ctx.neb_params:
            self.ctx.input_dict["MOTION"]["BAND"]["OPTIMIZE_BAND"][
                "OPTIMIZE_END_POINTS"
            ] = self.ctx.neb_params["optimize_end_points"]

        # resources
        self.ctx.options = self.inputs.options
        if self.inputs.dft_params["protocol"] == "debug":
            self.ctx.options = {
                "max_wallclock_seconds": 600,
                "resources": {
                    "num_machines": 3,
                    "num_mpiprocs_per_machine": 1,
                    "num_cores_per_mpiproc": 1,
                },
            }
        self.ctx.scf_options = deepcopy(self.ctx.options)
        # numper of mpi processes for scf derived from nproc_replica
        self.ctx.scf_options["resources"]["num_machines"] = int(
            self.ctx.neb_param["nproc_rep"]
            / self.ctx.options["resources"]["num_mpiprocs_per_machine"]
        )
        self.ctx.input_dict["GLOBAL"]["WALLTIME"] = max(
            600, self.ctx.options["max_wallclock_seconds"] - 600
        )
        # --------------------------------------------------

    def should_run_scf(self):
        """Function that returns whether to run or not the first scf step"""
        return self.ctx.should_run_scf

    def first_scf(self):
        """Run scf on the initial geometry."""

        files, input_dict, structure_with_tags = get_dft_inputs(
            self.inputs.dft_params.get_dict(),
            self.inputs.structure,
            "scf_ot_protocol.yml",
        )

        builder = Cp2kCalculation.get_builder()
        builder.structure = StructureData(ase=structure_with_tags)
        builder.code = self.inputs.code
        builder.file = files

        # resources
        builder.metadata.options = self.ctx.scf_options

        # label
        builder.metadata.label = "scf"

        # parser
        builder.metadata.options.parser_name = "cp2k_advanced_parser"

        # walltime
        input_dict["GLOBAL"]["WALLTIME"] = 86000

        # cp2k input dictionary
        builder.parameters = Dict(input_dict)

        future = self.submit(builder)
        self.report(f"Submitted scf of the initial geometry: {future.pk}")
        self.to_context(initial_scf=future)

    def submit_neb(self):
        self.report("Submitting NEB optimization")
        if not self.ctx.initial_scf.is_finished_ok:
            return self.exit_codes.ERROR_TERMINATION

        builder = Cp2kCalculation.get_builder()
        # code
        builder.code = self.inputs.code
        # structure
        builder.structure = StructureData(ase=self.ctx.structure_with_tags)
        builder.file = self.ctx.files
        # resources
        builder.metadata.options = self.ctx.options
        # parser
        builder.metadata.options.parser_name = "nanotech_empa.cp2k_neb_parser"
        # additional retrieved files
        builder.settings = Dict(
            dict={"additional_retrieve_list": ["*.xyz", "*.out", "*.ener"]}
        )

        # wfn cp commands.
        if self.ctx.should_run_scf:
            cp_commands = ""
            ndigits = len(str(self.ctx.neb_params["number_of_replica"]))
            for i in range(self.ctx.neb_params["number_of_replica"]):
                wfn_name = "aiida-BAND%s-RESTART.wfn" % str(i + 1).zfill(ndigits)
                cp_commands += (
                    "cp "
                    + self.ctx.initial_scf.outputs.remote_folder.get_remote_path()
                    + "/aiida-RESTART.wfn "
                    + "./"
                    + wfn_name
                    + "\n"
                )
        else:
            cp_commands = ""
            for wfn_cp_command in self.ctx.wfn_cp_commands:
                cp_commands += wfn_cp_command + "\n"

        builder.metadata.options.prepend_text = cp_commands

        # cp2k input dictionary
        builder.parameters = Dict(self.ctx.input_dict)

        future = self.submit(builder)
        self.to_context(neb=future)

    def finalize(self):
        self.report("Finalizing.")

        if not self.ctx.neb.is_finished_ok:
            return self.exit_codes.ERROR_TERMINATION

        for i_rep in range(self.ctx.neb_params["number_of_replica"]):
            label = "opt_replica_%s" % str(i_rep).zfill(3)
            self.out(label, self.ctx.neb.outputs[label])

        self.out("replica_energies", self.ctx.neb.outputs["replica_energies"])
        self.out("replica_distances", self.ctx.neb.outputs["replica_distances"])

        # Add the workchain pk to the input structure extras
        common_utils.add_extras(self.inputs.structure, "surfaces", self.node.uuid)

        return ExitCode(0)
