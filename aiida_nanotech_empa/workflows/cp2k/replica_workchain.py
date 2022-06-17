"""Workflow to run Replica Chain calculations with CP2K."""
from aiida import orm
from aiida.plugins import DataFactory, WorkflowFactory
import math

StructureData = DataFactory("structure")
Cp2kBaseWorkChain = DataFactory("cp2k.base")

class Cp2kReplicaWorkChain(WorkChain):
    """Workflow to run Replica Chain calculations with CP2K."""

    @classmethod
    def define(cls, spec):
        """Define the workflow."""

        # Define the inputs of the workflow
        spec.input("structure", valid_type=StructureData)
        spec.input("code", valid_type=orm.Code)
        spec.input("constraints", valid_type=orm.Str)
        spec.input("colvars", valid_type=orm.Str)
        spec.input("colvar_targets", valid_type=orm.List)
        spec.input("colvar_increments", valid_type=orm.List)

        spec.outline(
            cls.start,
            cls.first_scf,
            while(cls.should_run_simulations)(
                cls.update_colvar_increments,
                cls.run_constrained_geo_opts,
                cls.update_latest_structure,
            )
        )
    def start(self):
        """Initialise the work chain process."""
        self.ctx.latest_structure = self.inputs.structure
        self.ctx.colvars = self._get_actual_colvars()
        self.propagation_step = 0
    
    def first_scf(self):
        """Run the geometry optimization."""
        builder = Cp2kBaseWorkChain.get_builder()
        builder.structure = self.ctx.latest_structure
        builder.code = self.inputs.code
        submitted_calculation = self.submit(builder)
        self.report(f"Submitted initial geometry optimization: {submitted_node}")
        return ToContext(initial_geo_opt=submitted_calculation)
    
    def _get_actual_colvars(self):
        ase_structure = self.ctx.structure.get_ase()
        colvars = self.inputs.colvars.split()
        actual_colvars = []
        # ToDo: Check if the colvars are actually present in the structure.
        # ToDo: compute the actual colvars.
        return actual_colvars
        
    def update_colvar_increments(self):
        """Increment the colvars. If the target value is reached, set increment to 0."""
        self.ctx.colvar_increments = []
        for index, colvar in enumerate(self._get_actual_colvars()):
            if math.abs(spec.input.colvar_targets[index] - colvar) < self.inputs.increment_colvars[i]:
                self.ctx.colvar_increments.append(self.inputs.increment_colvars[i])
            else:
                self.ctx.colvar_increments.append(0)
    

    def run_constrained_geo_opts(self):
        """Run a constrained geometry optimizations."""
        builder = Cp2kBaseWorkChain.get_builder()
        builder.structure = self.ctx.latest_structure
        builder.code = self.inputs.code


        for index, value in enumerate(self._get_actual_colvars):
            if self.ctx.colvar_increments[index] == 0:
                continue

            # Populate the input dictionary with the constraints and colvars.
            builder.input_dict = {"colvar": 2}
            submitted_calculation = self.submit(builder)
            self.report(f"Submitted initial geometry optimization: {submitted_node}")
            self.to_context(**{f"run_{self.propagation_step}":append_(future)})

    
    def update_latest_structure(self):
        """Increment the colvars."""
        results = []
        for index, calculation in enumerate(getattr(self.ctx, f"run_{self.propagation_step}")):
            results.append((calculation.outputs.total_energy, index))
        
        lowest_energy_calc = sort(results, key=lambda x: x[0])[0][1]
        self.ctx.latest_structure = getattr(self.ctx, f"run_{self.propagation_step}")[lowest_energy_calc].outputs.structure
        self.ctx.propagation_step += 1
    
        