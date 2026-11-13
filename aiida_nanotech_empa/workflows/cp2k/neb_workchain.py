import copy
from numbers import Integral

import numpy as np
from aiida import engine, orm, plugins

from ...utils import common_utils
from . import cp2k_utils

Cp2kCalculation = plugins.CalculationFactory("cp2k")


def _initial_scf_resources(resources, nproc_rep, scheduler):
    """Fit one replica's MPI ranks within the BAND node/per-node limits."""
    for name, value in {"nproc_rep": nproc_rep, **resources}.items():
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    band = scheduler.create_job_resource(**resources)
    total = band.get_tot_num_mpiprocs()
    if nproc_rep > total or total % nproc_rep:
        raise ValueError("nproc_rep must divide the total BAND MPI allocation")

    # Use the fewest nodes that can hold exactly NPROC_REP ranks. Rounding up
    # the node count alone could silently run too many ranks (e.g. 8 on 2x6).
    first = (
        nproc_rep + band.num_mpiprocs_per_machine - 1
    ) // band.num_mpiprocs_per_machine
    for machines in range(first, min(band.num_machines, nproc_rep) + 1):
        if nproc_rep % machines == 0:
            break
    else:
        raise ValueError("Cannot fit nproc_rep within the BAND node/per-node limits")
    scf = copy.deepcopy(resources)
    scf["num_machines"] = machines
    scf["num_mpiprocs_per_machine"] = nproc_rep // machines
    if "tot_num_mpiprocs" in scf:
        scf["tot_num_mpiprocs"] = nproc_rep
    if "num_cores_per_machine" in scf:
        threads, remainder = divmod(
            scf["num_cores_per_machine"], band.num_mpiprocs_per_machine
        )
        if remainder or threads != scf.get("num_cores_per_mpiproc", threads):
            raise ValueError(
                "BAND cores per machine must match MPI ranks times threads"
            )
        scf["num_cores_per_machine"] = scf["num_mpiprocs_per_machine"] * threads
    scheduler.create_job_resource(**scf)
    return scf


class Cp2kNebWorkChain(engine.WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("code", valid_type=orm.Code)
        spec.input("structure", valid_type=orm.StructureData)
        spec.input_namespace(
            "replicas",
            valid_type=orm.StructureData,
            required=False,
            help="nodes of input replicas",
        )
        spec.input("wfn_cp_commands", valid_type=orm.Str, required=False)
        spec.input("restart_from", valid_type=orm.Str, required=False)
        spec.input(
            "protocol",
            valid_type=orm.Str,
            default=lambda: orm.Str("standard"),
            required=False,
            help="Protocol supported by the Cp2kDiagWorkChain.",
        )
        spec.input("dft_params", valid_type=orm.Dict)
        spec.input("sys_params", valid_type=orm.Dict)
        spec.input("neb_params", valid_type=orm.Dict)
        spec.input(
            "options",
            valid_type=dict,
            non_db=True,
            help="Define options for the cacluations: walltime, memory, CPUs, etc.",
        )

        # Workchain outline.
        spec.inputs.validator = cp2k_utils.validate_poisson_inputs
        spec.outline(
            cls.setup,
            engine.if_(cls.should_run_scf)(cls.first_scf),
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
            391,
            "ERROR_INCOMPLETE_NEB",
            message="NEB incomplete; available partial results are retained for inspection.",
        )
        spec.exit_code(
            392,
            "ERROR_INVALID_SCF_RESOURCES",
            message="Cannot allocate the initial SCF for one NEB replica environment.",
        )
        spec.exit_code(
            380,
            "ERROR_UUIDS",
            message="no structures specified",
        )

    def setup(self):
        self.report("Inspecting input and setting up things")
        if "restart_from" in self.inputs:
            dft_params = orm.load_node(
                self.inputs.restart_from.value
            ).inputs.dft_params.get_dict()
        else:
            dft_params = self.inputs.dft_params.get_dict()

        (
            self.ctx.files,
            self.ctx.input_dict,
            self.ctx.structure_with_tags,
        ) = cp2k_utils.get_dft_inputs(
            dft_params,
            self.inputs.structure,
            "neb_protocol.yml",
            self.inputs.protocol.value,
        )
        self.ctx.sys_params = self.inputs.sys_params.get_dict()
        self.ctx.neb_params = self.inputs.neb_params.get_dict()

        # The input structure is the  NEB replica 0 if we do not continue from previous NEB.
        all_replica_nodes = [self.inputs.structure]

        # Check if restarting.
        if "replicas" in self.inputs:
            for i in range(1, len(self.inputs.replicas) + 1):
                name = f"replica_{str(i).zfill(3)}"
                all_replica_nodes.append(self.inputs.replicas[name])
        elif "restart_from" in self.inputs:
            self.report("checking wfn available from previos neb")
            workcalc = orm.load_node(self.inputs.restart_from.value)
            n_previous_replica = workcalc.inputs.neb_params["number_of_replica"]
            all_replica_nodes = [
                workcalc.outputs[f"opt_replica_{str(i).zfill(3)}"]
                for i in range(n_previous_replica)
            ]
        else:
            return self.exit_codes.ERROR_UUIDS

        # Check for existing wfn files and create copy commands.
        self.ctx.should_run_scf = False
        if "wfn_cp_commands" in self.inputs:
            self.ctx.wfn_cp_commands = self.inputs.wfn_cp_commands.value
        else:
            self.ctx.wfn_cp_commands = cp2k_utils.mk_wfn_cp_commands(
                nreplicas=self.ctx.neb_params["number_of_replica"],
                replica_nodes=all_replica_nodes,
                selected_computer=self.inputs.code.computer,
                dft_params=dft_params,
            )

        if len(self.ctx.wfn_cp_commands) == 0:
            self.ctx.should_run_scf = True

        # Replica files with tags must be after structure_with_tags.
        self.ctx.input_dict["MOTION"]["BAND"]["REPLICA"] = []
        tags = self.ctx.structure_with_tags.get_tags().astype(np.int32).tolist()

        for i, node in enumerate(all_replica_nodes):
            filename = f"replica_{str(i).zfill(3)}" + ".xyz"
            self.ctx.files[filename.replace(".", "_")] = cp2k_utils.make_geom_file(
                node.get_ase(), filename, tags=tags
            )
            # Update the input dictionary.
            self.ctx.input_dict["MOTION"]["BAND"]["REPLICA"].append(
                {"COORD_FILE_NAME": filename}
            )

        # Constraints.
        if "constraints" in self.ctx.sys_params:
            self.ctx.input_dict["MOTION"]["CONSTRAINT"] = (
                cp2k_utils.get_constraints_section(self.ctx.sys_params["constraints"])
            )
        # Colvars.
        if "colvars" in self.ctx.sys_params:
            self.ctx.input_dict["FORCE_EVAL"]["SUBSYS"].update(
                cp2k_utils.get_colvars_section(self.ctx.sys_params["colvars"])
            )

        # NEB parameters.
        for param in [
            "align_frames",
            "rotate_frames",
            "band_type",
            "k_spring",
            "nproc_rep",
            "number_of_replica",
        ]:
            if param in self.ctx.neb_params:
                self.ctx.input_dict["MOTION"]["BAND"][param.upper()] = (
                    self.ctx.neb_params[param]
                )

        if "nsteps_it" in self.ctx.neb_params:
            self.ctx.input_dict["MOTION"]["BAND"]["CI_NEB"] = {
                "NSTEPS_IT": self.ctx.neb_params["nsteps_it"]
            }
        if "optimize_end_points" in self.ctx.neb_params:
            self.ctx.input_dict["MOTION"]["BAND"]["OPTIMIZE_BAND"][
                "OPTIMIZE_END_POINTS"
            ] = self.ctx.neb_params["optimize_end_points"]

        # Resources
        self.ctx.options = self.inputs.options
        self.ctx.scf_options = copy.deepcopy(self.ctx.options)

        if self.ctx.should_run_scf:
            try:
                self.ctx.scf_options["resources"] = _initial_scf_resources(
                    self.ctx.options["resources"],
                    self.ctx.neb_params["nproc_rep"],
                    self.inputs.code.computer.get_scheduler(),
                )
            except (KeyError, TypeError, ValueError) as exc:
                self.report(f"Invalid initial SCF resources: {exc}")
                return self.exit_codes.ERROR_INVALID_SCF_RESOURCES
        self.ctx.input_dict["GLOBAL"]["WALLTIME"] = max(
            600, self.ctx.options["max_wallclock_seconds"] - 600
        )

    def should_run_scf(self):
        """Function that returns whether to run or not the first scf step"""
        return self.ctx.should_run_scf

    def first_scf(self):
        """Run scf on the initial geometry."""

        files, input_dict, structure_with_tags = cp2k_utils.get_dft_inputs(
            self.inputs.dft_params.get_dict(),
            self.inputs.structure,
            "scf_ot_protocol.yml",
            self.inputs.protocol.value,
        )

        builder = Cp2kCalculation.get_builder()
        builder.structure = orm.StructureData(ase=structure_with_tags)
        builder.code = self.inputs.code
        builder.file = files
        builder.metadata.options = self.ctx.scf_options
        builder.metadata.label = "scf"
        builder.metadata.options.parser_name = "cp2k_advanced_parser"
        input_dict["GLOBAL"]["WALLTIME"] = max(
            600, self.ctx.options["max_wallclock_seconds"] - 600
        )
        builder.parameters = orm.Dict(input_dict)

        future = self.submit(builder)
        self.report(f"Submitted scf of the initial geometry: {future.pk}")
        self.to_context(initial_scf=future)

    def submit_neb(self):
        self.report("Submitting NEB optimization")
        if self.ctx.should_run_scf and not self.ctx.initial_scf.is_finished_ok:
            return self.exit_codes.ERROR_TERMINATION

        builder = Cp2kCalculation.get_builder()
        builder.code = self.inputs.code
        builder.structure = orm.StructureData(ase=self.ctx.structure_with_tags)
        builder.file = self.ctx.files
        builder.metadata.options = self.ctx.options
        builder.metadata.options.parser_name = "nanotech_empa.cp2k_neb_parser"
        builder.settings = orm.Dict(
            dict={"additional_retrieve_list": ["*.xyz", "*.out", "*.ener"]}
        )

        # wfn cp commands.
        if self.ctx.should_run_scf:
            cp_commands = ""
            ndigits = len(str(self.ctx.neb_params["number_of_replica"]))
            for i in range(self.ctx.neb_params["number_of_replica"]):
                wfn_name = f"aiida-BAND{str(i + 1).zfill(ndigits)}-RESTART.wfn"
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
        builder.parameters = orm.Dict(self.ctx.input_dict)
        self.to_context(neb=self.submit(builder))

    def finalize(self):
        self.report("Finalizing.")

        # Publish available results even when native convergence/time limits fail.
        # The GUI and restart path must retain access to an incomplete band.
        for label in self.ctx.neb.outputs:
            if label.startswith("opt_replica_") or label in (
                "replica_energies",
                "replica_distances",
                "remote_folder",
                "output_parameters",
            ):
                self.out(label, self.ctx.neb.outputs[label])

        common_utils.add_extras(self.inputs.structure, "surfaces", self.node.uuid)
        if not self.ctx.neb.is_finished_ok:
            if "replica_energies" in self.ctx.neb.outputs:
                return self.exit_codes.ERROR_INCOMPLETE_NEB
            return self.exit_codes.ERROR_TERMINATION
        return engine.ExitCode(0)
