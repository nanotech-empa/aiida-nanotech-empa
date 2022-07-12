import numpy as np
import pathlib
import yaml
from aiida import engine, orm, plugins
from aiida_nanotech_empa.utils import analyze_structure
from aiida_nanotech_empa.workflows.cp2k import cp2k_utils

StructureData = plugins.DataFactory('structure')
Cp2kBaseWorkChain = plugins.WorkflowFactory('cp2k.base')

DATA_DIR = pathlib.Path(__file__).parent.absolute() / 'data'


def load_protocol(fname, protocol):
    """Load a protocol from a file."""
    with open(pathlib.Path(__file__).parent / 'protocols' / fname,
              encoding="utf-8") as fhandle:
        protocols = yaml.safe_load(fhandle)
        return protocols[protocol]


@engine.calcfunction
def total_charge(charges):
    return orm.Int(sum(charges))


#@engine.calcfunction
def split_structure(structure, fixed_atoms, magnetization_per_site, fragments):
    ase_geo = structure.get_ase()

    allfixed = [0 for i in ase_geo]
    mps = []
    fixed = ''

    yield {
        'label': "total",
        'structure': structure,
        'fixed_atoms': fixed_atoms,
        'magnetization_per_site': magnetization_per_site,
    }

    for f in analyze_structure.string_range_to_list(fixed_atoms.value,
                                                    shift=-1)[0]:
        allfixed[f] = 1

    for fragment_label, fragment in fragments.items():
        fragment = sorted(fragment)

        if magnetization_per_site or fixed_atoms.value:
            tuples = [(e, *np.round(p, 2))
                      for e, p in zip(ase_geo[fragment].get_chemical_symbols(),
                                      ase_geo[fragment].positions)]
            if magnetization_per_site:
                mps = [
                    m for at, m in zip(ase_geo, list(magnetization_per_site))
                    if (at.symbol, *np.round(at.position, 2)) in tuples
                ]
                if all(m == 0 for m in mps):
                    mps = []
            if fixed_atoms.value:
                fixed = [
                    f for at, f in zip(ase_geo, allfixed)
                    if (at.symbol, *np.round(at.position, 2)) in tuples
                ]
                fixed = analyze_structure.list_to_string_range(
                    np.nonzero(fixed)[0].tolist(), shift=1)
        yield {
            'label': fragment_label,
            'structure': StructureData(ase=ase_geo[fragment]),
            'fixed_atoms': orm.Str(fixed),
            'magnetization_per_site': orm.List(list=mps),
        }


@engine.calcfunction
def calc_energies(energies, whole_ene):
    """Calculate adsorption energies."""
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

    return orm.Dict(dict=adsorption_energies)


class Cp2kFragmentSeparationWorkChain(engine.WorkChain):
    """WorkChain to compute adsoprtion energy for a molecule on a substrate."""
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("code", valid_type=orm.Code)

        # Specify the input structure and its fragments.
        spec.input("structure",
                   valid_type=StructureData,
                   help="A molecule on a substrate.")

        spec.input_namespace(
            "fragments",
            valid_type=orm.List,
            help="Fragments of the system to be treated individually.")

        # Charges of each fragment.
        spec.input_namespace(
            "charges",  # +1 means one electron removed
            valid_type=orm.Int,
            help=
            "Charges of each fragment. No need to specify the total charge as it would be computed automatically."
        )

        # Total multiplicity of the system and of the fragments.
        spec.input_namespace(
            "multiplicities",
            valid_type=orm.Int,
            #default=lambda: orm.Int(0),
            help=
            "Multiplicity of each fragment. Use 'total' to specify the multiplicity of the whole system."
        )

        # Fixed atoms and magnetization per site defined for the whole system. Will be automatically split for the fragments.
        spec.input(
            "fixed_atoms",
            valid_type=orm.Str,
            #default=lambda: orm.List(list=[]),
            help="Fixed atoms of the system.")

        spec.input(
            "magnetization_per_site",
            valid_type=orm.List,
            #default=lambda: orm.List(list=[]),
            help="Magnetization per site.")

        # Protocol that defines the simulation settings.
        spec.input(
            "protocol",
            valid_type=orm.Str,
            #default=lambda: orm.Str('standard'),
            required=False,
            help="Settings to run simulations with.")

        spec.input_namespace(
            "auxilary_dictionaries",
            valid_type=orm.Dict,
            help=
            "Dictionaries to append to the CP2K input dictionary right before the job submission. "
            "It is useful to add constraints and collective variables to the input."
        )

        spec.input_namespace(
            "options",
            valid_type=dict,
            non_db=True,
            required=False,
            help=
            "Define options for the cacluations: walltime, memory, CPUs, etc.")

        # Outline.
        spec.outline(cls.setup, cls.run_scfs, cls.run_geo_opts, cls.finalize)

        # Dynamic outputs.
        spec.outputs.dynamic = True

        # TODO: double-check the exit codes.
        # Exit code.
        spec.exit_code(
            380,
            "ERROR_SUBSTRATE_NOT_SUPPORTED",
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
        """Setup the work chain."""

        self.report("Inspecting input and setting things up.")

        n_atoms = len(self.inputs.structure.get_ase())
        n_mags = len(list(self.inputs.magnetization_per_site))
        if n_mags not in (0, n_atoms):
            self.report(
                "If set, magnetization_per_site needs a value for every atom.")
            return self.exit_codes.ERROR_TERMINATION  # pylint: disable=no-member

        # Check if the substrate is supported.
        # if len(an_out) != len(list(self.inputs.fragments)) and len(an_out) < 2:
        #     self.report('Structure analyis failed.')
        #     return self.exit_codes.ERROR_STRUCTURE_ANALYSIS

        self.ctx.cutoff = cp2k_utils.get_cutoff(
            structure=self.inputs.structure)

        self.ctx.file = {
            'basis': orm.SinglefileData(file=DATA_DIR / "BASIS_MOLOPT"),
            'pseudo': orm.SinglefileData(file=DATA_DIR / "POTENTIAL"),
            'dftd3':
            orm.SinglefileData(file=DATA_DIR /
                               "dftd3.dat"),  # TODO: make vdw optional
        }

        return engine.ExitCode(0)

    def run_scfs(self):
        """Run SCF calculation for the whole system and for fragments."""
        # Firt run SCF for the whole system and its fragments.
        for inputs in split_structure(
                structure=self.inputs.structure,
                fixed_atoms=self.inputs.fixed_atoms,
                magnetization_per_site=self.inputs.magnetization_per_site,
                fragments=self.inputs.fragments,
        ):

            # Re-loading the input dictionary for the given protocol.
            input_dict = load_protocol(fname="slab_opt_protocol.yml",
                                       protocol=self.inputs.protocol.value)

            # Fragment's label.
            fragment = inputs['label']

            # Generic inputs that are always the same.
            builder = Cp2kBaseWorkChain.get_builder()
            builder.cp2k.code = self.inputs.code
            builder.cp2k.metadata.options = self.inputs.options[fragment]
            builder.cp2k.file = self.ctx.file
            input_dict['GLOBAL']['RUN_TYPE'] = 'ENERGY'
            input_dict['FORCE_EVAL']['DFT']['MGRID'][
                'CUTOFF'] = self.ctx.cutoff
            builder.cp2k.metadata.options.parser_name = "cp2k_advanced_parser"

            if "max_wallclock_seconds" in self.inputs.options[fragment]:
                # Calculation might require up to 5 minutes to gracefully finish the calculation.
                input_dict['GLOBAL']['WALLTIME'] = max(
                    self.inputs.options[fragment]["max_wallclock_seconds"] -
                    300, 300)

            # Dealing with magnetization.
            structure_with_tags, kinds_dict = cp2k_utils.determine_kinds(
                inputs["structure"], inputs["magnetization_per_site"])
            builder.cp2k.structure = structure_with_tags
            kinds_section = cp2k_utils.get_kinds_section(kinds_dict,
                                                         protocol='gpw')
            # Append the new kinds section to CP2K input dictionary.
            cp2k_utils.dict_merge(
                input_dict, kinds_section
            )

            # If multiplicity is set and it is greater than 0, switch the UKS on.
            if "multiplicities" in self.inputs and fragment in self.inputs.multiplicities and self.inputs.multiplicities[
                    fragment].value > 0:
                input_dict['FORCE_EVAL']['DFT']['UKS'] = '.TRUE.'
                input_dict['FORCE_EVAL']['DFT'][
                    'MULTIPLICITY'] = self.multiplicities[fragment].value

            # If charge is set and it is greater than 0, add it to the corresponding section of the iput.
            if "charges" in self.inputs and fragment in self.inputs.charges and self.inputs.charges[
                    fragment].value != 0:
                input_dict['FORCE_EVAL']['DFT']['CHARGE'] = self.charges[
                    fragment].value

            # Fixed atoms
            input_dict['MOTION']['CONSTRAINT']['FIXED_ATOMS']['LIST'] = inputs[
                "fixed_atoms"]

            # Finally, append auxilary dictionaries to the input dictonary.
            if 'auxilary_dictionaries' in self.inputs:
                for _, value in self.inputs.auxilary_dictionaries.items():
                    cp2k_utils.dict_merge(input_dict, value)

            builder.cp2k.parameters = orm.Dict(dict=input_dict)

            submitted_node = self.submit(builder)
            self.to_context(**{f"scf.{fragment}": submitted_node})

    def run_geo_opts(self):
        for fragment in self.inputs.fragments.keys():
            # We deliberately do not run optimisation for the total structure.
            if fragment == "total":
                continue

            # Generic inputs that are always the same.
            builder = Cp2kBaseWorkChain.get_builder()
            builder.cp2k.code = self.inputs.code
            builder.cp2k.metadata.options = self.inputs.options[fragment]
            builder.cp2k.file = self.ctx.file
            builder.cp2k.metadata.options.parser_name = "cp2k_advanced_parser"

            # Re-loading the input dictionary for the given protocol.
            previous_calc = self.ctx[f"scf.{fragment}"]
            input_dict = previous_calc.inputs.cp2k.parameters.get_dict()
            input_dict['GLOBAL']['RUN_TYPE'] = 'GEO_OPT'

            builder.cp2k.parameters = orm.Dict(dict=input_dict)
            builder.cp2k.parent_calc_folder = previous_calc.outputs.remote_folder
            builder.cp2k.structure = previous_calc.inputs.cp2k.structure

            submitted_node = self.submit(builder)
            self.to_context(**{f"opt.{fragment}": submitted_node})

    def finalize(self):
        energies = {}
        self.report("Finalizing...")
        energies["total"] = self.ctx["scf.total"].outputs.output_parameters[
            'energy']

        separation_energy = energies["total"]
        unrelaxed_separation_energy = energies["total"]

        for fragment in list(self.inputs.fragments.keys()):
            energies[fragment] = {
                'unrelaxed':
                self.ctx[f"scf.{fragment}"].outputs.
                output_parameters['energy'],
                'relaxed':
                self.ctx[f"opt.{fragment}"].outputs.output_parameters['energy']
            }
            separation_energy -= energies[fragment]['relaxed']
            unrelaxed_separation_energy -= energies[fragment]['unrelaxed']

        energies["unrelaxed_separation_energy"] = unrelaxed_separation_energy
        energies["separation_energy"] = separation_energy

        energies = orm.Dict(dict=energies).store()
        self.out('energies', energies)

        # Add the workchain pk to the input structure extras
        extras_label = "Cp2kAdsorptionEnergyWorkChain_pks"
        if extras_label not in self.inputs.structure.extras:
            extras_list = []
        else:
            extras_list = self.inputs.structure.extras[extras_label]
        extras_list.append(self.node.pk)
        self.inputs.structure.set_extra(extras_label, extras_list)
