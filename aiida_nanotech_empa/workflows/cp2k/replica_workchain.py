"""Workflow to run Replica Chain calculations with CP2K."""
from aiida import orm
from aiida.plugins import DataFactory, WorkflowFactory
import math
from nanotech_empa.workflows.cp2k.cp2k_utils import compute_colvars

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
        spec.input("colvars_targets", valid_type=orm.List)
        spec.input("colvars_increments", valid_type=orm.List)
        spec.input("continuation_of", valid_type=orm.Int)

        spec.outline(
            cls.start,
            cls.first_scf,
            cls.update_colvars_values,
            cls.update_colvars_increments,
            while(cls.should_run_simulations)(
                cls.run_constrained_geo_opts,
                cls.update_latest_structure,
                cls.update_colvars_values,
                cls.update_colvars_increments,
            )
        )
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
        self.report(f"Submitted scf of the initial geometry: {submitted_node}")
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
            if math.abs(spec.input.colvars_targets[index] - colvar) < self.inputs.colvars_increments[i]:
                self.ctx.colvars_increments.append(math.abs(self.inputs.colvars_increments[i])*np.sign(self.inputs.colvar_targets[i] - colvar))
            else:
                self.ctx.colvars_increments.append(0)
        if all(i = 0 for i in self.ctx.colvar_increments):
            self.ctx.should_run_simulations = False
    

    def run_constrained_geo_opts(self):
        """Run a constrained geometry optimization for each non 0 increment of colvars."""
        builder = Cp2kBaseWorkChain.get_builder()
        builder.structure = self.ctx.latest_structure
        builder.code = self.inputs.code


        for index, value in enumerate(self.ctx.colvars_values):
            if self.ctx.colvars_increments[index] == 0:
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
    
        