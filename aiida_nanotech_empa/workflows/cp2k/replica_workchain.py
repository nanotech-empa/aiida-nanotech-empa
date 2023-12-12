import math

import numpy as np
from aiida import engine, orm, plugins

from ...utils import common_utils
from . import cp2k_utils

Cp2kBaseWorkChain = plugins.WorkflowFactory("cp2k.base")
Cp2kCalculation = plugins.CalculationFactory("cp2k")


class Cp2kReplicaWorkChain(engine.WorkChain):
    """Workflow to run Replica Chain calculations with CP2K."""

    @classmethod
    def define(cls, spec):
        """Define the workflow."""
        super().define(spec)

        # Define the inputs of the workflow.
        spec.input("code", valid_type=orm.Code)
        spec.input("structure", valid_type=orm.StructureData)
        spec.input("parent_calc_folder", valid_type=orm.RemoteData, required=False)
        spec.input("restart_from", valid_type=orm.Str, required=False)
        spec.input(
            "protocol",
            valid_type=orm.Str,
            default=lambda: orm.Str("standard"),
            required=False,
            help="Protocol supported by the Cp2kBaseWorkChain.",
        )
        spec.input("dft_params", valid_type=orm.Dict)
        spec.input("sys_params", valid_type=orm.Dict)
        spec.input(
            "options",
            valid_type=dict,
            non_db=True,
            help="Define options for the cacluations: walltime, memory, CPUs, etc.",
        )

        spec.outline(
            cls.setup,
            engine.if_(cls.should_run_scf)(
                cls.first_scf,
                cls.update_colvars_values,
                cls.update_colvars_increments,
                cls.to_outputs,
            ),
            engine.while_(cls.should_run_simulations)(
                cls.run_constrained_geo_opts,
                cls.update_latest_structure,
                cls.update_colvars_values,
                cls.update_colvars_increments,
                cls.to_outputs,
            ),
            cls.finalize,
        )

        spec.outputs.dynamic = True
        spec.output_namespace("structures", valid_type=orm.StructureData)
        spec.output_namespace("details", valid_type=orm.Dict)
        spec.exit_code(390, "ERROR_TERMINATION", message="One geo opt failed")

    def setup(self):
        """Initialize the workchain process."""
        self.report("Inspecting input and setting up things")

        # Restart from previous workchain
        if "restart_from" in self.inputs:
            self.report(
                f"Retrieving previous steps from {self.inputs.restart_from.value} and continuing"
            )
            previous_replica = orm.load_node(self.inputs.restart_from.value)
            previous_structures = list(previous_replica.outputs.structures)
            previous_structures.sort()

            self.ctx.lowest_energy_structure = previous_replica.outputs.structures[
                previous_structures[-1]
            ]
            for struc in previous_structures:
                self.out(
                    f"structures.{struc}", previous_replica.outputs.structures[struc]
                )
                self.out(f"details.{struc}", previous_replica.outputs.details[struc])
            self.ctx.CVs_to_increment = previous_replica.outputs.details[struc][
                "cvs_target"
            ]
            self.ctx.colvars_values = previous_replica.outputs.details[struc][
                "cvs_actual"
            ]
            self.ctx.should_run_scf = False
            self.ctx.propagation_step = (
                len(previous_structures) - 1
            )  # continue form this step
            self.update_colvars_increments()
            self.ctx.should_run_simulations = True
            self.ctx.restart_folder = list(
                self.ctx.lowest_energy_structure.get_incoming(
                    node_class=orm.CalcJobNode
                )
            )[-1][0].outputs.remote_folder
            self.report(f"data from workchain: {previous_replica.pk}")
            self.report(f"actual CVs: {self.ctx.colvars_values}")
            self.report(f"CVs to increment: {self.ctx.CVs_to_increment}")
            self.report(
                f"starting from geometry: {self.ctx.lowest_energy_structure.pk}"
            )
            self.report(f"retrieved the following steps: {previous_structures} ")
        else:
            self.ctx.lowest_energy_structure = self.inputs.structure
            self.ctx.should_run_scf = True
            self.ctx.should_run_simulations = True
            self.ctx.propagation_step = 0
        return engine.ExitCode(0)

    def should_run_scf(self):
        """Function that returnns whether to run or not the first scf step"""
        return self.ctx.should_run_scf

    def should_run_simulations(self):
        """Function that returnns whether targets have been reached or not"""
        return self.ctx.should_run_simulations

    def to_outputs(self):
        """Function to update step by step the workcain output"""
        if self.ctx.propagation_step == 0:
            self.out(
                "details.initial_scf",
                orm.Dict(
                    dict={
                        "output_parameters": dict(
                            self.ctx.initial_scf.outputs.output_parameters
                        ),
                        "cvs_target": self.ctx.colvars_values,
                        "cvs_actual": self.ctx.colvars_values,
                        "d2prev": 0,
                    }
                ).store(),
            )
            self.out("structures.initial_scf", self.ctx.lowest_energy_structure)
            self.report("Updated output for the initial_scf step")
        else:
            self.out(
                f"details.step_{self.ctx.propagation_step - 1 :04}",
                orm.Dict(
                    dict={
                        "output_parameters": dict(
                            self.ctx.lowest_energy_output_parameters
                        ),
                        "cvs_target": self.ctx.CVs_cases[self.ctx.lowest_energy_calc],
                        "cvs_actual": self.ctx.colvars_values,
                        "d2prev": self.ctx.d2prev,
                    }
                ).store(),
            )
            self.out(
                f"structures.step_{self.ctx.propagation_step - 1 :04}",
                self.ctx.lowest_energy_structure,
            )
            self.report(f"Updated output for step {self.ctx.propagation_step - 1 :04}")
        return engine.ExitCode(0)

    def first_scf(self):
        """Run scf on the initial geometry."""

        files, input_dict, structure_with_tags = cp2k_utils.get_dft_inputs(
            self.inputs.dft_params.get_dict(),
            self.ctx.lowest_energy_structure,
            "scf_ot_protocol.yml",
            self.inputs.protocol.value,
        )

        builder = Cp2kCalculation.get_builder()
        builder.structure = orm.StructureData(ase=structure_with_tags)
        builder.code = self.inputs.code
        builder.file = files
        if "parent_calc_folder" in self.inputs:
            builder.parent_calc_folder = self.inputs.parent_calc_folder
        builder.metadata.options = self.inputs.options
        builder.metadata.label = "scf"
        builder.metadata.options.parser_name = "cp2k_advanced_parser"
        input_dict["GLOBAL"]["WALLTIME"] = max(
            600, self.inputs.options["max_wallclock_seconds"] - 600
        )
        builder.parameters = orm.Dict(input_dict)

        future = self.submit(builder)
        self.report(f"Submitted SCF of the initial geometry: {future.pk}")
        self.to_context(initial_scf=future)

    def update_colvars_values(self):
        """Compute actual value of CVs."""

        ase_structure = self.ctx.lowest_energy_structure.get_ase()
        colvars = self.inputs.sys_params["colvars"]
        self.ctx.colvars_values = [
            cv[1] for cv in cp2k_utils.compute_colvars(colvars, ase_structure)
        ]
        self.report(f"actual CVs values: {self.ctx.colvars_values}")
        if self.ctx.propagation_step == 0:
            self.ctx.CVs_to_increment = self.ctx.colvars_values
        else:
            self.ctx.CVs_to_increment = self.ctx.CVs_cases[self.ctx.lowest_energy_calc]

    def update_colvars_increments(self):
        """Computes the increments for the CVs according to deviation from target.
        If the target value is reached wihtin the increment, set increment to 0.
        Deviation from target is computed wrt actual value of CVs while new CVs
        are computed as previous target plus increment to avoid slow diverging deviations
        from targets"""
        self.ctx.colvars_increments = []
        for index, colvar in enumerate(self.ctx.colvars_values):
            if (
                math.fabs(self.inputs.sys_params["colvars_targets"][index] - colvar)
                > math.fabs(self.inputs.sys_params["colvars_increments"][index])
                and math.fabs(self.inputs.sys_params["colvars_increments"][index])
                > 0.001
            ):
                self.ctx.colvars_increments.append(
                    math.fabs(self.inputs.sys_params["colvars_increments"][index])
                    * np.sign(self.inputs.sys_params["colvars_targets"][index] - colvar)
                )
            else:
                self.ctx.colvars_increments.append(0)
        if all(i == 0 for i in self.ctx.colvars_increments):
            self.ctx.should_run_simulations = False

    def run_constrained_geo_opts(self):
        """Run a constrained geometry optimization for each non 0 increment of colvars."""
        self.ctx.CVs_cases = []
        for index in range(len(self.ctx.CVs_to_increment)):
            if self.ctx.colvars_increments[index] != 0:
                structure = self.ctx.lowest_energy_structure

                files, input_dict, structure_with_tags = cp2k_utils.get_dft_inputs(
                    self.inputs.dft_params,
                    structure,
                    "geo_opt_protocol.yml",
                    self.inputs.protocol.value,
                )

                builder = Cp2kBaseWorkChain.get_builder()
                builder.cp2k.code = self.inputs.code
                builder.cp2k.structure = orm.StructureData(ase=structure_with_tags)
                builder.cp2k.file = files
                builder.cp2k.metadata.options = self.inputs.options
                builder.cp2k.metadata.options.parser_name = "cp2k_advanced_parser"
                builder.handler_overrides = orm.Dict(
                    {"restart_incomplete_calculation": True}
                )
                if self.ctx.propagation_step == 0:
                    builder.cp2k.parent_calc_folder = (
                        self.ctx.initial_scf.outputs.remote_folder
                    )
                else:
                    builder.cp2k.parent_calc_folder = self.ctx.restart_folder

                if "constraints" in self.inputs.sys_params:
                    input_dict["MOTION"][
                        "CONSTRAINT"
                    ] = cp2k_utils.get_constraints_section(
                        self.inputs.sys_params["constraints"]
                    )
                if "colvars" in self.inputs.sys_params:
                    input_dict["FORCE_EVAL"]["SUBSYS"].update(
                        cp2k_utils.get_colvars_section(
                            self.inputs.sys_params["colvars"]
                        )
                    )
                # Update constraints.
                submitted_cvs = ""
                current_cvs_targets = []
                for icv, cvval in enumerate(self.ctx.CVs_to_increment):
                    target = cvval
                    units = input_dict["MOTION"]["CONSTRAINT"]["COLLECTIVE"][icv][
                        "TARGET"
                    ].split(" ")[0]
                    if icv == index:
                        target += self.ctx.colvars_increments[icv]
                    current_cvs_targets.append(target)
                    input_dict["MOTION"]["CONSTRAINT"]["COLLECTIVE"][icv]["TARGET"] = (
                        units + " " + str(target)
                    )
                    submitted_cvs += " " + str(target)
                self.ctx.CVs_cases.append(current_cvs_targets)
                input_dict["GLOBAL"]["WALLTIME"] = max(
                    600, self.inputs.options["max_wallclock_seconds"] - 600
                )
                builder.cp2k.parameters = orm.Dict(input_dict)

                submitted_calculation = self.submit(builder)
                self.report(
                    f"Submitted GEO OPT: {submitted_calculation.pk}, with {submitted_cvs}"
                )
                self.to_context(
                    **{
                        f"run_{self.ctx.propagation_step :04}": engine.append_(
                            submitted_calculation
                        )
                    }
                )

    def update_latest_structure(self):
        """Update the latest structure as the one with minimum energy from the constrained
        geometry optimizations."""
        results = []
        for index, calculation in enumerate(
            getattr(self.ctx, f"run_{self.ctx.propagation_step :04}")
        ):
            # check if the calculation is finished
            if not common_utils.check_if_calc_ok(self, calculation):
                return self.exit_codes.ERROR_TERMINATION
            results.append((calculation.outputs.output_parameters["energy_scf"], index))
        self.report(f"Energies: {results}")
        results.sort(key=lambda x: x[0])
        self.ctx.lowest_energy_calc = results[0][1]
        lowest_energy_base_workchain = getattr(
            self.ctx, f"run_{self.ctx.propagation_step :04}"
        )[self.ctx.lowest_energy_calc]
        ase_previous = self.ctx.lowest_energy_structure.get_ase()
        self.ctx.lowest_energy_structure = (
            lowest_energy_base_workchain.outputs.output_structure
        )
        ase_now = self.ctx.lowest_energy_structure.get_ase()
        self.ctx.d2prev = np.linalg.norm(ase_previous.positions - ase_now.positions)
        self.ctx.lowest_energy_output_parameters = (
            lowest_energy_base_workchain.outputs.output_parameters
        )
        self.ctx.lowest_energy = results[0][0]
        self.report(
            f"The lowest energy at step {self.ctx.propagation_step :04} is: {self.ctx.lowest_energy}"
        )
        self.report(f"geometry: {self.ctx.lowest_energy_structure.pk}")
        self.report(f"target CVs {self.ctx.CVs_cases[self.ctx.lowest_energy_calc]}")
        self.ctx.restart_folder = getattr(
            self.ctx, f"run_{self.ctx.propagation_step :04}"
        )[self.ctx.lowest_energy_calc].outputs.remote_folder
        self.ctx.propagation_step += 1
        return engine.ExitCode(0)

    def finalize(self):
        self.report("Finalizing...")

        # Add the workchain uuid to the input structure extras.
        common_utils.add_extras(self.inputs.structure, "surfaces", self.node.uuid)
        return engine.ExitCode(0)
