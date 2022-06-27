from aiida_nanotech_empa.utils import common_utils

import numpy as np

from aiida.engine import WorkChain, ExitCode, calcfunction, ToContext, if_
from aiida.orm import Int, Str, Bool, Code, List, Dict
from aiida.orm import StructureData

from aiida.plugins import WorkflowFactory

Cp2kMoleculeGwWorkChain = WorkflowFactory('nanotech_empa.cp2k.molecule_gw')
Cp2kMoleculeOptWorkChain = WorkflowFactory('nanotech_empa.cp2k.molecule_opt')


@calcfunction
def analyze_structure(structure, mag_per_site):

    mol_atoms = structure.get_ase()

    mps = []
    if list(mag_per_site):
        mol_at_tuples = [(e, *np.round(p, 2)) for e, p in zip(
            mol_atoms.get_chemical_symbols(), mol_atoms.positions)]
        mps = [
            m for at, m in zip(mol_atoms, list(mag_per_site))
            if (at.symbol, *np.round(at.position, 2)) in mol_at_tuples
        ]

    return {
        'mol_struct': StructureData(ase=mol_atoms),
        'mol_mag_per_site': List(list=mps),
    }


class Cp2kMoleculeOptGwWorkChain(WorkChain):
    """
    WorkChain to  optimize molecule and run GW 

    Two different ways to run:
    1) optimize geo and run gw
    2) run gw
    """
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("code", valid_type=Code)

        spec.input("structure",
                   valid_type=StructureData,
                   help="An isolated molecule.")
        spec.input("protocol",
                   valid_type=Str,
                   default=lambda: Str('gpw_std'),
                   required=False,
                   help="Protocol supported by the GW workchain.")
        spec.input("multiplicity",
                   valid_type=Int,
                   default=lambda: Int(0),
                   required=False)
        spec.input("magnetization_per_site",
                   valid_type=List,
                   default=lambda: List(list=[]),
                   required=False)
        spec.input("walltime_seconds",
                   valid_type=Int,
                   default=lambda: Int(600),
                   required=False)
        spec.input("max_nodes",
                   valid_type=Int,
                   default=lambda: Int(2056),
                   required=False)
        spec.input("gw_options",
                   valid_type=Dict,
                   default=lambda: Dict(
                       dict={
                           'resources': {
                               'num_mpiprocs_per_machine': 1,
                               'num_cores_per_mpiproc': 6,
                           }
                       }))
        spec.input("debug",
                   valid_type=Bool,
                   default=lambda: Bool(False),
                   required=False,
                   help="Run with fast parameters for debugging.")
        spec.input("geo_opt",
                   valid_type=Bool,
                   default=lambda: Bool(True),
                   required=False,
                   help="Perform geo opt step.")
        spec.input("geometry_opt_mult",
                   valid_type=Int,
                   default=lambda: Int(0),
                   required=False,
                   help="Multiplicity in case of 'gas opt' is selected.")

        spec.outline(cls.setup,
                     if_(cls.gas_opt_selected)(cls.gas_opt, cls.check_gas_opt),
                     cls.gw, cls.finalize)
        spec.outputs.dynamic = True

        spec.exit_code(
            390,
            "ERROR_TERMINATION",
            message="One or more steps of the work chain failed.",
        )

    def setup(self):
        self.report("Inspecting input and setting up things.")

        n_atoms = len(self.inputs.structure.get_ase())
        n_mags = len(list(self.inputs.magnetization_per_site))
        if n_mags not in (0, n_atoms):
            self.report(
                "If set, magnetization_per_site needs a value for every atom.")
            return self.exit_codes.ERROR_TERMINATION  # pylint: disable=no-member

        an_out = analyze_structure(self.inputs.structure,
                                   self.inputs.magnetization_per_site)

        self.ctx.mol_struct = an_out['mol_struct']
        self.ctx.mol_mag_per_site = an_out['mol_mag_per_site']

        return ExitCode(0)

    def gas_opt_selected(self):
        return self.inputs.geo_opt.value

    def gas_opt(self):
        builder = Cp2kMoleculeOptWorkChain.get_builder()
        builder.code = self.inputs.code
        builder.structure = self.ctx.mol_struct
        builder.multiplicity = self.inputs.geometry_opt_mult
        builder.magnetization_per_site = self.ctx.mol_mag_per_site
        builder.vdw = Bool(True)
        builder.walltime_seconds = self.inputs.walltime_seconds
        builder.debug = self.inputs.debug
        builder.metadata.description = "Submitted by Cp2kMoleculeOptGwWorkChain."
        builder.metadata.label = 'Cp2kMoleculeOptWorkChain'
        submitted_node = self.submit(builder)
        return ToContext(gas_opt=submitted_node)

    def check_gas_opt(self):
        if not common_utils.check_if_calc_ok(self, self.ctx.gas_opt):
            return self.exit_codes.ERROR_TERMINATION  # pylint: disable=no-member
        # set the optimized geometry as ctx geometry

        self.ctx.mol_struct = self.ctx.gas_opt.outputs.output_structure
        return ExitCode(0)

    def gw(self):

        self.report("Submitting GW.")

        builder = Cp2kMoleculeGwWorkChain.get_builder()
        builder.code = self.inputs.code
        builder.protocol = self.inputs.protocol
        builder.structure = self.ctx.mol_struct
        builder.magnetization_per_site = self.ctx.mol_mag_per_site
        builder.multiplicity = self.inputs.multiplicity
        builder.debug = self.inputs.debug
        builder.walltime_seconds = self.inputs.walltime_seconds
        builder.max_nodes = self.inputs.max_nodes
        builder.options = self.inputs.gw_options
        builder.metadata.description = "gw"
        submitted_node = self.submit(builder)
        return ToContext(gw=submitted_node)

    def finalize(self):
        self.report("Finalizing...")

        if not common_utils.check_if_calc_ok(self, self.ctx.gw):
            return self.exit_codes.ERROR_TERMINATION  # pylint: disable=no-member

        gw_out_params = self.ctx.gw.outputs.gw_output_parameters
        self.out('gw_output_parameters', gw_out_params)

        self.out('output_structure', self.ctx.mol_struct)
        # Add the workchain pk to the input/geo_opt structure extras

        struc_to_label = self.ctx.mol_struct
        extras_label = "Cp2kMoleculeOptGwWorkChain_pks"
        if extras_label not in struc_to_label.extras:
            extras_list = []
        else:
            extras_list = struc_to_label.extras[extras_label]
        extras_list.append(self.node.pk)
        struc_to_label.set_extra(extras_label, extras_list)

        return ExitCode(0)
