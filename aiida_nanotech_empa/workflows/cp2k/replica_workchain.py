from aiida_nanotech_empa.workflows.cp2k.cp2k_utils import (
    get_colvars_section,
    get_constraints_section,
    get_dft_inputs,
)
from aiida_nanotech_empa.utils import common_utils
from aiida_nanotech_empa.workflows.cp2k.cp2k_utils import compute_colvars
from aiida.plugins import DataFactory, WorkflowFactory, CalculationFactory
from aiida.engine import (
    WorkChain,
    ExitCode,
    while_,
    if_,
    append_,
    # calcfunction,
)
from aiida.orm import Str, Code, Dict, load_node, CalcJobNode, RemoteData
import math
import numpy as np

StructureData = DataFactory("core.structure")
Cp2kBaseWorkChain = WorkflowFactory("cp2k.base")
Cp2kCalculation = CalculationFactory("cp2k")


# @calcfunction
# def output_dict(enes, cvs, structures):
#
#    return Dict({"energies": enes, "cvs": listcvs, "structures": structures})


class Cp2kReplicaWorkChain(WorkChain):
    """Workflow to run Replica Chain calculations with CP2K."""

    @classmethod
    def define(cls, spec):
        """Define the workflow."""
        super().define(spec)

        # Define the inputs of the workflow
        spec.input("code", valid_type=Code)
        spec.input("structure", valid_type=StructureData)
        spec.input("parent_calc_folder", valid_type=RemoteData, required=False)
        spec.input("restart_from", valid_type=Str, required=False)
        spec.input("dft_params", valid_type=Dict)
        spec.input("sys_params", valid_type=Dict)
        spec.input(
            "options",
            valid_type=dict,
            non_db=True,
            help="Define options for the cacluations: walltime, memory, CPUs, etc.",
        )

        spec.outline(
            cls.setup,
            if_(cls.should_run_scf)(
                cls.first_scf,
                cls.update_colvars_values,
                cls.update_colvars_increments,
                cls.to_outputs,
            ),
            while_(cls.should_run_simulations)(
                cls.run_constrained_geo_opts,
                cls.update_latest_structure,
                cls.update_colvars_values,
                cls.update_colvars_increments,
                cls.to_outputs,
            ),
            cls.finalize,
        )

        spec.outputs.dynamic = True
        spec.output_namespace("structures", valid_type=StructureData)
        spec.output_namespace("details", valid_type=Dict)
        spec.exit_code(390, "ERROR_TERMINATION", message="One geo opt failed")

    def setup(self):
        """Initialize the workchain process."""
        self.report("Inspecting input and setting up things")

        # restart from previous workchain
        if "restart_from" in self.inputs:
            self.report(
                f"Retrieving previous steps from {self.inputs.restart_from.value} and continuing"
            )
            previous_replica = load_node(self.inputs.restart_from.value)
            previous_structures = [
                struc for struc in previous_replica.outputs.structures
            ]

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
                self.ctx.lowest_energy_structure.get_incoming(node_class=CalcJobNode)
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
        return ExitCode(0)

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
                Dict(
                    dict={
                        "output_parameters": dict(
                            self.ctx.initial_scf.outputs.output_parameters
                        ),
                        "cvs_target": self.ctx.colvars_values,
                        "cvs_actual": self.ctx.colvars_values,
                    }
                ).store(),
            )
            self.out("structures.initial_scf", self.ctx.lowest_energy_structure)
            self.report("Updated output for the initial_scf step")
        else:
            self.out(
                f"details.step_{self.ctx.propagation_step - 1 :04}",
                Dict(
                    dict={
                        "output_parameters": dict(
                            self.ctx.lowest_energy_output_parameters
                        ),
                        "cvs_target": self.ctx.CVs_cases[self.ctx.lowest_energy_calc],
                        "cvs_actual": self.ctx.colvars_values,
                    }
                ).store(),
            )
            self.out(
                f"structures.step_{self.ctx.propagation_step - 1 :04}",
                self.ctx.lowest_energy_structure,
            )
            self.report(f"Updated output for step {self.ctx.propagation_step - 1 :04}")
        return ExitCode(0)

    def first_scf(self):
        """Run scf on the initial geometry."""

        files, input_dict, structure_with_tags = get_dft_inputs(
            self.inputs.dft_params.get_dict(),
            self.ctx.lowest_energy_structure,
            "scf_ot_protocol.yml",
        )

        builder = Cp2kCalculation.get_builder()
        builder.structure = StructureData(ase=structure_with_tags)
        builder.code = self.inputs.code
        builder.file = files
        # restart wfn
        if "parent_calc_folder" in self.inputs:
            builder.parent_calc_folder = self.inputs.parent_calc_folder

        # resources
        builder.metadata.options = self.inputs.options

        # label
        builder.metadata.label = "scf"

        # parser
        builder.metadata.options.parser_name = "cp2k_advanced_parser"

        # cp2k input dictionary
        builder.parameters = Dict(input_dict)

        future = self.submit(builder)
        self.report(f"Submitted scf of the initial geometry: {future.pk}")
        self.to_context(initial_scf=future)

    def update_colvars_values(self):
        """Compute actual value of CVs."""

        ase_structure = self.ctx.lowest_energy_structure.get_ase()
        colvars = self.inputs.sys_params["colvars"]
        self.ctx.colvars_values = [
            cv[1] for cv in compute_colvars(colvars, ase_structure)
        ]
        self.report(f"actual CVs values: {self.ctx.colvars_values}")
        if self.ctx.propagation_step == 0:
            self.ctx.CVs_to_increment = self.ctx.colvars_values
        else:
            self.ctx.CVs_to_increment = self.ctx.CVs_cases[self.ctx.lowest_energy_calc]
            # self.report(f"will add increments to this set of CVs: {self.ctx.CVs_to_increment}")

    def update_colvars_increments(self):
        """Computes teh increments for the CVs according to deviation from target.
        If the target value is reached wihtin the increment, set increment to 0.
        Deviation from target is computed wrt actual value of CVs while new CVs
        are computed as previous target plus increment to avoid slow diverging deviations
        from targets"""
        self.ctx.colvars_increments = []
        for index, colvar in enumerate(self.ctx.colvars_values):
            if (
                math.fabs(self.inputs.sys_params["colvars_targets"][index] - colvar)
                > self.inputs.sys_params["colvars_increments"][index]
                and math.fabs(self.inputs.sys_params["colvars_increments"][index])
                > 0.0001
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
        # pylint: disable=unused-variable
        self.ctx.CVs_cases = []
        for index, value in enumerate(
            self.ctx.CVs_to_increment
        ):  # (self.ctx.colvars_values):
            if self.ctx.colvars_increments[index] != 0:
                structure = self.ctx.lowest_energy_structure

                files, input_dict, structure_with_tags = get_dft_inputs(
                    self.inputs.dft_params, structure, "geo_opt_protocol.yml"
                )

                builder = Cp2kBaseWorkChain.get_builder()
                builder.cp2k.code = self.inputs.code
                builder.cp2k.structure = StructureData(ase=structure_with_tags)
                builder.cp2k.file = files
                # resources
                builder.cp2k.metadata.options = self.inputs.options
                # parser
                builder.cp2k.metadata.options.parser_name = "cp2k_advanced_parser"
                # handlers
                builder.handler_overrides = Dict(
                    {"restart_incomplete_calculation": True}
                )
                # wfn restart folder
                if self.ctx.propagation_step == 0:
                    builder.cp2k.parent_calc_folder = (
                        self.ctx.initial_scf.outputs.remote_folder
                    )
                else:
                    builder.cp2k.parent_calc_folder = self.ctx.restart_folder

                # constraints.
                if "constraints" in self.inputs.sys_params:
                    input_dict["MOTION"]["CONSTRAINT"] = get_constraints_section(
                        self.inputs.sys_params["constraints"]
                    )
                # colvars.
                if "colvars" in self.inputs.sys_params:
                    input_dict["FORCE_EVAL"]["SUBSYS"].update(
                        get_colvars_section(self.inputs.sys_params["colvars"])
                    )
                # update constraints
                submitted_CVs = ""
                current_CVs_targets = []
                for icv, cvval in enumerate(self.ctx.CVs_to_increment):
                    target = cvval
                    units = input_dict["MOTION"]["CONSTRAINT"]["COLLECTIVE"][icv][
                        "TARGET"
                    ].split(" ")[0]
                    if icv == index:
                        target += self.ctx.colvars_increments[icv]
                    current_CVs_targets.append(target)
                    input_dict["MOTION"]["CONSTRAINT"]["COLLECTIVE"][icv]["TARGET"] = (
                        units + " " + str(target)
                    )
                    submitted_CVs += " " + str(target)
                self.ctx.CVs_cases.append(current_CVs_targets)

                # cp2k input dictionary
                builder.cp2k.parameters = Dict(input_dict)

                submitted_calculation = self.submit(builder)
                self.report(
                    f"Submitted geo opt: {submitted_calculation.pk}, with {submitted_CVs}"
                )
                self.to_context(
                    **{
                        f"run_{self.ctx.propagation_step :04}": append_(
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
                return self.exit_codes.ERROR_TERMINATION  # pylint: disable=no-member
            results.append((calculation.outputs.output_parameters["energy_scf"], index))
        self.report(f"energies {results}")
        results.sort(key=lambda x: x[0])  # pylint: disable=expression-not-assigned
        self.ctx.lowest_energy_calc = results[0][1]
        lowest_energy_base_workchain = getattr(
            self.ctx, f"run_{self.ctx.propagation_step :04}"
        )[self.ctx.lowest_energy_calc]
        self.ctx.lowest_energy_structure = (
            lowest_energy_base_workchain.outputs.output_structure
        )
        self.ctx.lowest_energy_output_parameters = (
            lowest_energy_base_workchain.outputs.output_parameters
        )
        self.ctx.lowest_energy = results[0][0]
        self.report(
            f"The lowest energy at step {self.ctx.propagation_step :04} is {self.ctx.lowest_energy}"
        )
        self.report(f"geometry: {self.ctx.lowest_energy_structure.pk}")
        self.report(f"target CVs {self.ctx.CVs_cases[self.ctx.lowest_energy_calc]}")
        # define restart folder
        self.ctx.restart_folder = getattr(
            self.ctx, f"run_{self.ctx.propagation_step :04}"
        )[self.ctx.lowest_energy_calc].outputs.remote_folder

        # increment step index
        self.ctx.propagation_step += 1
        return ExitCode(0)

    def finalize(self):
        self.report("Finalizing...")
        # self.out('output_parameters',Dict(dict={'energies':self.ctx.outenes,'cvs':self.ctx.outcvs,'structures':self.ctx.outstructures}).store())

        # Add the workchain pk to the input structure extras
        common_utils.add_extras(self.inputs.structure, "surfaces", self.node.uuid)
        return ExitCode(0)
