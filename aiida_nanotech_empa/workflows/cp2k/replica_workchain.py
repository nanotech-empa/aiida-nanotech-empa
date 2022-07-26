"""Workflow to run Replica Chain calculations with CP2K."""
from aiida.plugins import DataFactory
from aiida.engine import WorkChain, ToContext, ExitCode, while_, append_
from aiida_nanotech_empa.utils import common_utils
from aiida_nanotech_empa.workflows.cp2k.cp2k_utils import compute_colvars
from aiida.orm import Int, Str, Code, List, Bool
import math
import numpy as np

StructureData = DataFactory("structure")
Cp2kBaseWorkChain = DataFactory("cp2k.base")
#Cp2kOptWorkChain = WorkflowFactory('nanotech_empa.cp2k.opt')


class Cp2kReplicaWorkChain(WorkChain):
    """Workflow to run Replica Chain calculations with CP2K."""
    @classmethod
    def define(cls, spec):
        """Define the workflow."""

        # Define the inputs of the workflow
        spec.input(
            "charge",  # +1 means one electron removed
            valid_type=Int,
            default=lambda: Int(0),
            required=False)
        spec.input("multiplicity",
                   valid_type=Int,
                   default=lambda: Int(0),
                   required=False)
        spec.input("magnetization_per_site",
                   valid_type=List,
                   default=lambda: List(list=[]),
                   required=False)
        spec.input("vdw",
                   valid_type=Bool,
                   default=lambda: Bool(False),
                   required=False)
        spec.input("protocol",
                   valid_type=Str,
                   default=lambda: Str('standard'),
                   required=False,
                   help="Settings to run simulations with.")
        spec.input("structure", valid_type=StructureData)
        spec.input("code", valid_type=Code)
        spec.input("constraints", valid_type=Str)
        spec.input("colvars", valid_type=Str)
        spec.input("colvars_targets", valid_type=List)
        spec.input("colvars_increments", valid_type=List)
        spec.input("continuation_of",
                   valid_type=Int,
                   default=lambda: Int(0),
                   required=False)

        spec.outline(
            cls.start, cls.first_scf, cls.update_colvars_values,
            cls.update_colvars_increments,
            while_(cls.should_run_simulations)(
                cls.run_constrained_geo_opts,
                cls.update_latest_structure,
                cls.update_colvars_values,
                cls.update_colvars_increments,
            ))

        spec.exit_code(390, "ERROR_TERMINATION", message="One geo opt failed")

    def start(self):
        """Initialise the workchain process."""
        self.ctx.latest_structure = self.inputs.structure
        self.ctx.should_run_simulations = True
        #self.ctx.colvars = self._get_actual_colvars()
        self.propagation_step = 0

    def first_scf(self):
        """Run scf on the initial geometry."""
        builder = Cp2kBaseWorkChain.get_builder()
        builder.structure = self.ctx.latest_structure
        builder.code = self.inputs.code
        submitted_calculation = self.submit(builder)
        self.report(
            f"Submitted scf of the initial geometry: {submitted_calculation}")
        return ToContext(initial_scf=submitted_calculation)

    def update_colvars_values(self):
        ase_structure = self.ctx.latest_structure.get_ase()
        colvars = self.inputs.colvars.value
        self.ctx.colvars_values = compute_colvars(colvars, ase_structure)

    def update_colvars_increments(self):
        """Increment/decrement the colvars according to deviation from target. 
        If the target value is reached, set increment to 0."""
        self.ctx.colvar_increments = []
        for index, colvar in enumerate(self.ctx.colvars_values):
            if math.abs(self.inputs.colvars_targets[index] -
                        colvar) < self.inputs.colvars_increments[index]:
                self.ctx.colvars_increments.append(
                    math.abs(self.inputs.colvars_increments[index]) *
                    np.sign(self.inputs.colvar_targets[index] - colvar))
            else:
                self.ctx.colvars_increments.append(0)
        if all(i == 0 for i in self.ctx.colvar_increments):
            self.ctx.should_run_simulations = False

    def run_constrained_geo_opts(self):
        """Run a constrained geometry optimization for each non 0 increment of colvars."""
        builder = Cp2kBaseWorkChain.get_builder()
        builder.structure = self.ctx.latest_structure
        builder.code = self.inputs.code
        #pylint: disable=unused-variable
        for index, value in enumerate(self.ctx.colvars_values):
            if self.ctx.colvars_increments[index] != 0:
                targets_to_use = np.array(self.ctx.colvars_values)
                targets_to_use[index] += self.ctx.colvars_increments[index]

                # Populate the input dictionary with the constraints and colvars.
                builder.input_dict = {"colvar": 2}
                submitted_calculation = self.submit(builder)
                self.report(
                    f"Submitted initial geometry optimization: {submitted_calculation}"
                )
                future = 'something'
                self.to_context(
                    **{f"run_{self.propagation_step}": append_(future)})

    def update_latest_structure(self):
        """Update the latest structure as teh one with minimum energy from the constrained 
        geometry optimizations."""
        results = []
        for index, calculation in enumerate(
                getattr(self.ctx, f"run_{self.propagation_step}")):
            #check if the calculation is finished
            if not common_utils.check_if_calc_is_ok(calculation):
                return self.exit_codes.ERROR_TERMINATION  # pylint: disable=no-member
            results.append((calculation.outputs.total_energy, index))
        results.sort(key=lambda x: x[0])[0][1]  #pylint: disable=expression-not-assigned
        lowest_energy_calc = results[0][1]
        self.ctx.latest_structure = getattr(
            self.ctx, f"run_{self.propagation_step}"
        )[lowest_energy_calc].outputs.structure
        self.report(
            f"The lowest energy at step {self.ctx.propagation_step} is {results[0][0]}"
        )
        self.ctx.propagation_step += 1
        return ExitCode(0)
