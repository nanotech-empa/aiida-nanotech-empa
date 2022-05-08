from aiida_nanotech_empa.utils import common_utils

import numpy as np

from aiida.engine import WorkChain, ExitCode, calcfunction  #, ToContext  #, if_
from aiida.orm import Int, Bool, Code, List, Str, Dict, Float
from aiida.orm import StructureData
from aiida_nanotech_empa.utils import analyze_structure

from aiida.plugins import WorkflowFactory

Cp2kStdOptWorkChain = WorkflowFactory('nanotech_empa.cp2k.std_opt')
Cp2kStdScfWorkChain = WorkflowFactory('nanotech_empa.cp2k.std_scf')
Cp2kBsseWorkChain = WorkflowFactory('nanotech_empa.cp2k.bsse')

#def geometrical_analysis(ase_geo):
#    """Simple geometry analysis that returns in the case of
#    1) an isolated molecule -> geometry, None
#   2) adsorbed system -> molecular geometry, top substr. layer z
#   """
#   chem_symbols_arr = np.array(ase_geo.get_chemical_symbols())
#   s_atoms = ase_geo[chem_symbols_arr == substr_elem]
#   non_s_atoms = ase_geo[chem_symbols_arr != substr_elem]
#   if len(s_atoms) == 0:
#       return ase_geo, None
#   layer_dz = 1.0  # ang
#   substr_top_layer = s_atoms[np.abs(
#       np.max(s_atoms.positions[:, 2]) - s_atoms.positions[:, 2]) < layer_dz]
#   surf_z = np.mean(substr_top_layer.positions[:, 2])
#
#   mol_atoms = non_s_atoms[non_s_atoms.positions[:, 2] > surf_z]
#
#   return mol_atoms, surf_z


@calcfunction
def total_charge(charges):
    return Int(sum(charges))


@calcfunction
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
def split_structure(structure, charges, fixed_atoms, fragments, multiplicity,
                    mag_per_site):
    ase_geo = structure.get_ase()

    return_dict = {}
    allfixed = [0 for i in ase_geo]
    mps = []
    fix = ''
    charges = list(charges)
    if not multiplicity:
        multiplicity = [0 for f in fragments]
    for f in analyze_structure.string_range_to_list(fixed_atoms.value,
                                                    shift=-1)[0]:
        allfixed[f] = 1
    for i, fragment in enumerate(fragments):
        fragment = analyze_structure.string_range_to_list(fragment,
                                                          shift=-1)[0]
        fragment.sort()
        charge = 0
        if charges:
            charge = charges[i]
        if mag_per_site or fixed_atoms.value:
            tuples = [(e, *np.round(p, 2))
                      for e, p in zip(ase_geo[fragment].get_chemical_symbols(),
                                      ase_geo[fragment].positions)]
            if mag_per_site:
                mps = [
                    m for at, m in zip(ase_geo, list(mag_per_site))
                    if (at.symbol, *np.round(at.position, 2)) in tuples
                ]
                if all(m == 0 for m in mps):
                    mps = []
            if fixed_atoms.value:
                fix = [
                    f for at, f in zip(ase_geo, allfixed)
                    if (at.symbol, *np.round(at.position, 2)) in tuples
                ]
                fix = analyze_structure.list_to_string_range(
                    np.nonzero(fix)[0].tolist(), shift=1)
        label = f"frag_{i}"
        return_dict[label] = {'struc': StructureData(ase=ase_geo[fragment])}
        return_dict[label]['charge'] = Int(charge)
        return_dict[label]['fix'] = Str(fix)
        return_dict[label]['mult'] = Int(multiplicity[i])
        return_dict[label]['mag'] = List(list=mps)

    return return_dict


@calcfunction
def calc_energies(energies, whole_ene):
    adsorption_energies = {}

    energies = dict(energies)
    sum_unrelax_ene = 0.0
    sum_relax_ene = 0.0
    for fragment in energies:
        deformation_ene = energies[fragment]['unrelaxed'] - energies[fragment][
            'relaxed']
        sum_unrelax_ene += energies[fragment]['unrelaxed']
        sum_relax_ene += energies[fragment]['relaxed']

        adsorption_energies[fragment] = {'deformation': deformation_ene}
    adsorption_energies['unrelaxed'] = whole_ene.value - sum_unrelax_ene
    adsorption_energies['relaxed'] = whole_ene.value - sum_relax_ene

    return Dict(dict=adsorption_energies)


class Cp2kAdsorptionEnergyWorkChain(WorkChain):
    """
    WorkChain to compute adsoprtion energy for a molecule on a substrate
    """
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("code", valid_type=Code)

        spec.input("structure",
                   valid_type=StructureData,
                   help="A molecule on a substrate.")
        spec.input(
            "charge",  # +1 means one electron removed
            valid_type=List,
            default=lambda: List(list=[]),
            required=False)
        spec.input("multiplicity",
                   valid_type=List,
                   default=lambda: List(list=[]),
                   required=False)
        spec.input("whole_multiplicity",
                   valid_type=Int,
                   default=lambda: Int(0),
                   required=False)
        spec.input("fixed_atoms",
                   valid_type=Str,
                   default=lambda: Str(''),
                   required=False)
        spec.input("magnetization_per_site",
                   valid_type=List,
                   default=lambda: List(list=[]),
                   required=False)
        spec.input("fragments",
                   valid_type=List,
                   default=lambda: List(list=[]),
                   required=False)
        spec.input("protocol",
                   valid_type=Str,
                   default=lambda: Str('standard'),
                   required=False,
                   help="Settings to run simulations with.")
        spec.input("walltime_seconds",
                   valid_type=Int,
                   default=lambda: Int(600),
                   required=False)
        spec.input("max_nodes",
                   valid_type=Int,
                   default=lambda: Int(48),
                   required=False)

        #spec.outline(cls.setup, cls.bsse, cls.frag_opt, cls.finalize)
        spec.outline(cls.setup, cls.scf, cls.frag_opt, cls.finalize)
        spec.outputs.dynamic = True

        spec.exit_code(
            380,
            "ERROR_SUBSTR_NOT_SUPPORTED",
            message="Specified substrate is not supported.",
        )
        spec.exit_code(
            381,
            "ERROR_STRUCTURE_ANALYSIS",
            message="Structure analysis failed.",
        )
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

        an_out = split_structure(self.inputs.structure, self.inputs.charge,
                                 self.inputs.fixed_atoms,
                                 self.inputs.fragments,
                                 self.inputs.multiplicity,
                                 self.inputs.magnetization_per_site)

        if len(an_out) != len(list(self.inputs.fragments)) and len(an_out) < 2:
            self.report('Structure analyis failed.')
            return self.exit_codes.ERROR_STRUCTURE_ANALYSIS

        self.ctx.fragments = an_out
        self.ctx.total_charge = total_charge(self.inputs.charge)

        return ExitCode(0)

    def scf(self):
        builder = Cp2kStdScfWorkChain.get_builder()
        builder.code = self.inputs.code
        builder.structure = self.inputs.structure
        builder.multiplicity = self.inputs.whole_multiplicity
        builder.charge = self.ctx.total_charge
        builder.magnetization_per_site = self.inputs.magnetization_per_site
        builder.vdw = Bool(True)
        builder.walltime_seconds = self.inputs.walltime_seconds
        builder.protocol = self.inputs.protocol
        label = 'scf_whole_system'
        submitted_node = self.submit(builder)
        #submitted_node.description = label
        self.to_context(**{label: submitted_node})

    def bsse(self):
        builder = Cp2kBsseWorkChain.get_builder()
        builder.code = self.inputs.code
        builder.structure = self.inputs.structure
        builder.fragments = self.inputs.fragments
        builder.whole_multiplicity = self.inputs.whole_multiplicity
        builder.multiplicity = self.inputs.multiplicity
        builder.charge = self.inputs.charge
        builder.magnetization_per_site = self.inputs.magnetization_per_site
        builder.vdw = Bool(True)
        builder.walltime_seconds = self.inputs.walltime_seconds
        builder.protocol = self.inputs.protocol
        label = 'BSSE'
        submitted_node = self.submit(builder)
        #submitted_node.description = label
        self.to_context(**{label: submitted_node})

    def frag_opt(self):
        for fragment in self.ctx.fragments:
            builder = Cp2kStdOptWorkChain.get_builder()
            builder.code = self.inputs.code
            builder.structure = self.ctx.fragments[fragment]['struc']
            builder.fixed_atoms = self.ctx.fragments[fragment]['fix']
            builder.multiplicity = self.ctx.fragments[fragment]['mult']
            builder.charge = self.ctx.fragments[fragment]['charge']
            builder.magnetization_per_site = self.ctx.fragments[fragment][
                'mag']
            builder.vdw = Bool(True)
            builder.walltime_seconds = self.inputs.walltime_seconds
            builder.protocol = self.inputs.protocol
            label = fragment + '_opt'
            submitted_node = self.submit(builder)
            #submitted_node.description = label
            self.to_context(**{label: submitted_node})
            #ToContext(**{label: submitted_node})
        #return

    def finalize(self):
        energies = {}
        self.report("Finalizing...")
        for fragment in self.ctx.fragments:
            label = fragment + '_opt'
            if not common_utils.check_if_calc_ok(self, getattr(
                    self.ctx, label)):
                return self.exit_codes.ERROR_TERMINATION  # pylint: disable=no-member

            calc = getattr(self.ctx, label)
            calc = dict(calc.outputs.output_parameters)
            energies[fragment] = {
                'unrelaxed': calc['motion_step_info']['energy_au'][0],
                'relaxed': calc['energy']
            }

        scf_ene = Float(
            self.ctx.scf_whole_system.outputs.output_parameters['energy'])
        self.out('energies', calc_energies(Dict(dict=energies), scf_ene))

        # Add the workchain pk to the input structure extras
        extras_label = "Cp2kAdsorptionEnergyWorkChain_pks"
        if extras_label not in self.inputs.structure.extras:
            extras_list = []
        else:
            extras_list = self.inputs.structure.extras[extras_label]
        extras_list.append(self.node.pk)
        self.inputs.structure.set_extra(extras_label, extras_list)

        return ExitCode(0)
