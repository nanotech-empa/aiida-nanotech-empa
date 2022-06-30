import numpy as np
import pathlib
import yaml
import copy
from aiida import engine, orm, plugins
from aiida_cp2k.utils import merge_dict, merge_Dict
from aiida_nanotech_empa import utils
from aiida_nanotech_empa.workflows.cp2k import cp2k_utils

StructureData = plugins.DataFactory('structure')
Cp2kBaseWorkChain = plugins.WorkflowFactory('cp2k.base')

DATA_DIR = pathlib.Path(__file__).parent.absolute() / 'data'


def load_protocol(fname, protocol):
    """Load a protocol from a file."""
    with open(pathlib.Path(__file__).parent / 'protocols' / fname) as fhandle:
        protocols = yaml.safe_load(fhandle)
        input_dict = copy.deepcopy(protocols[protocol])


@engine.calcfunction
def total_charge(charges):
    return Int(sum(charges))


@engine.calcfunction
def split_structure(structure, fragments, fixed_atoms, magnetisation_per_site):
    ase_geo = structure.get_ase()

    return_dict = {}
    allfixed = [0 for i in ase_geo]
    mps = []
    fix = ''

    yield {
        'label': label,
        'structure': structure,
        'fixed_atoms': fixed_atoms,
        'magnetization_per_site': magnetisation_per_site,
    }

    for f in utils.analyze_structure.string_range_to_list(fixed_atoms.value,
                                                          shift=-1)[0]:
        allfixed[f] = 1
    for i, fragment in enumerate(fragments):
        fragment = utils.analyze_structure.string_range_to_list(fragment,
                                                                shift=-1)[0]
        fragment.sort()

        if magnetisation_per_site or fixed_atoms.value:
            tuples = [(e, *np.round(p, 2))
                      for e, p in zip(ase_geo[fragment].get_chemical_symbols(),
                                      ase_geo[fragment].positions)]
            if magnetisation_per_site:
                mps = [
                    m for at, m in zip(ase_geo, list(magnetisation_per_site))
                    if (at.symbol, *np.round(at.position, 2)) in tuples
                ]
                if all(m == 0 for m in mps):
                    mps = []
            if fixed_atoms.value:
                fixed = [
                    f for at, f in zip(ase_geo, allfixed)
                    if (at.symbol, *np.round(at.position, 2)) in tuples
                ]
                fixed = utils.analyze_structure.list_to_string_range(
                    np.nonzero(fix)[0].tolist(), shift=1)
        label = f"fragment_{i}"
        yield {
            'label': label,
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
            "append_dict",
            valid_type=orm.Dict,
            help="Dictionary to append right before the job submission.")

        spec.input(
            "options",
            valid_type=dict,
            non_db=True,
            required=False,
            help="Define options for the cacluations: run time, memory, etc.")

        # Outline.
        spec.outline(cls.setup, cls.run_scfs)

        # Dynamic outputs.
        spec.outputs.dynamic = True

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

        # TODO: make sure all fragments have the same labels.

        self.ctx.cutoff = cp2k_utils.get_cutoff(
            structure=self.inputs.structure)

        self.ctx.file = {
            'basis': orm.SinglefileData(file=DATA_DIR / "BASIS_MOLOPT"),
            'pseudo': orm.SinglefileData(file=DATA_DIR / "POTENTIAL"),
            'dftd3': orm.SinglefileData(file=DATA_DIR / "dftd3.dat"),
        }

        return engine.ExitCode(0)

    def run_scfs(self):
        """Run SCF calculation for the whole system and for fragments."""

        input_dict = load_protocol(fname="fragment_separation.yml",
                                   protocol=self.inputs.protocol.value)

        # Firt run SCF for the whole system and its fragments.
        for inputs in split_structure(
                structure=self.inputs.structure,
                fixed_atoms=self.inputs.fixed_atoms,
                fragments=self.inputs.fragments,
                magnetisation_per_site=self.inputs.magnetization_per_site):
            fragment = inputs['fragment_label']

            builder = Cp2kBaseWorkChain.get_builder()
            builder.cp2k.code = self.inputs.code
            builder.cp2k.file = self.ctx.file

            # If multiplicity is set and it is greater than 0, switch the UKS on.
            if fragment in self.multiplicities and self.multiplicities[
                    fragment].value > 0:
                input_dict['FORCE_EVAL']['DFT']['UKS'] = '.TRUE.'
                input_dict['FORCE_EVAL']['DFT'][
                    'MULTIPLICITY'] = self.multiplicities[fragment].value

            # If charge is set and it is greater than 0, add it to the corresponding section of the iput.
            if fragment in self.charges and self.charges[fragment].value > 0:
                input_dict['FORCE_EVAL']['DFT']['CHARGE'] = self.charges[
                    fragment].value

            # Dealing with magnetisation.
            structure_with_tags, kinds_dict = cp2k_utils.determine_kinds(
                inputs["structure"], inputs["magnetization_per_site"])
            builder.structure = structure_with_tags

            for append_type, value in self.inputs.append_dict.items():
                merge_dict(input_dict, value)

            builder.cp2k.parameters = orm.Dict(dict=input_dict)

            # Walltime, memory, and resources.
            builder.cp2k.metadata.options = self.inputs.options

            submitted_node = self.submit(builder)
            self.to_context(**{fragment: submitted_node})

    def run_geo_opts(self):
        for fragment in self.ctx.fragments:
            builder = Cp2kStdOptWorkChain.get_builder()
            builder.code = self.inputs.code
            builder.structure = self.ctx.fragments[fragment]['structure']
            builder.fixed_atoms = self.ctx.fragments[fragment]['fixed_atoms']
            builder.multiplicity = self.ctx.fragments[fragment]['mult']
            builder.charge = self.ctx.fragments[fragment]['charge']
            builder.magnetization_per_site = self.ctx.fragments[fragment][
                'mag']
            builder.vdw = orm.Bool(True)
            builder.walltime_seconds = self.inputs.walltime_seconds
            builder.protocol = self.inputs.protocol
            label = fragment + '_opt'
            submitted_node = self.submit(builder)
            self.to_context(**{label: submitted_node})

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

        scf_energy = orm.Float(
            self.ctx.scf_whole_system.outputs.output_parameters['energy'])
        self.out('energies', calc_energies(orm.Dict(dict=energies),
                                           scf_energy))

        # Add the workchain pk to the input structure extras
        extras_label = "Cp2kAdsorptionEnergyWorkChain_pks"
        if extras_label not in self.inputs.structure.extras:
            extras_list = []
        else:
            extras_list = self.inputs.structure.extras[extras_label]
        extras_list.append(self.node.pk)
        self.inputs.structure.set_extra(extras_label, extras_list)
