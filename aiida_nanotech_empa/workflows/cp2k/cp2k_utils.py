# pylint: disable=too-many-locals
import pathlib
import yaml
import collections
import tempfile
import shutil
from io import StringIO
from aiida.orm import StructureData, Dict, SinglefileData

ang_2_bohr = 1.889725989


def get_kinds_section(kinds_dict, protocol='gapw_std'):
    """ Write the &KIND sections in gw calculations given the structure and the settings_dict"""

    bset = 'gapw_std_gw_basis_set'
    bsetaux = 'gapw_std_gw_basis_set_aux'
    potential = 'all'
    if protocol == 'gapw_hq':
        bset = 'gapw_hq_gw_basis_set'
        bsetaux = 'gapw_hq_gw_basis_set_aux'
        potential = 'all'
    elif protocol == 'gpw_std':
        bset = 'gpw_std_gw_basis_set'
        bsetaux = 'gpw_std_gw_basis_set_aux'
        potential = 'pseudopotential'
    elif protocol == 'gpw':
        bset = 'basis_set'
        bsetaux = ''
        potential = 'pseudopotential'
    kinds = []
    with open(pathlib.Path(__file__).parent / './data/atomic_kinds.yml',
              encoding='utf-8') as fhandle:
        atom_data = yaml.safe_load(fhandle)

    for kind_name in kinds_dict:
        element = ''.join([c for c in kind_name if not c.isdigit()])
        magnetization = kinds_dict[kind_name]['mag']
        is_ghost = kinds_dict[kind_name]['ghost']
        new_section = {
            '_': kind_name,
            'BASIS_SET': atom_data[bset][element],
            'POTENTIAL': atom_data[potential][element],
            'ELEMENT': element,
        }
        if bsetaux:
            new_section['BASIS_SET RI_AUX'] = atom_data[bsetaux][element]
        if is_ghost:
            new_section['GHOST'] = 'TRUE'
        if magnetization != 0.0:
            new_section['MAGNETIZATION'] = magnetization
        kinds.append(new_section)

    return {'FORCE_EVAL': {'SUBSYS': {'KIND': kinds}}}


def tags_and_magnetization(structure, magnetization_per_site):
    """Gather the same atoms with the same magnetization into one atomic kind."""
    ase_structure = structure.get_ase()
    if magnetization_per_site:
        if len(magnetization_per_site) != len(ase_structure.numbers):
            raise ValueError(
                'The size of `magnetization_per_site` is different from the number of atoms.'
            )

        # Combine atom type with magnetizations.
        complex_symbols = [
            f'{symbol}_{magn}' for symbol, magn in zip(
                ase_structure.get_chemical_symbols(), magnetization_per_site)
        ]
        # Assign a unique tag for every atom kind. do not use set in enumerate to avoid random order...!
        combined = {
            symbol: tag + 1
            for tag, symbol in enumerate(
                list(dict.fromkeys(complex_symbols).keys()))
        }
        # Assigning correct tags to every atom.
        tags = [combined[key] for key in complex_symbols]
        ase_structure.set_tags(tags)
        # Tag-magnetization correspondance.
        tags_correspondance = {
            str(value): float(key.split('_')[1])
            for key, value in combined.items()
        }
        return StructureData(ase=ase_structure), Dict(tags_correspondance)

    # we force tags to be 0 if magnetization vector is not provided, this ensures we do not get a structure with unnecessary labels

    tags = [0 for i in range(len(ase_structure))]
    ase_structure.set_tags(tags)
    return StructureData(ase=ase_structure), None


def determine_kinds(structure,
                    magnetization_per_site=None,
                    ghost_per_site=None):
    """Gather the same atoms with the same magnetization into one atomic kind."""
    ase_structure = structure.get_ase()

    if magnetization_per_site is None or len(magnetization_per_site) == 0:
        magnetization_per_site = [0 for i in range(len(ase_structure))]
    if ghost_per_site is None:
        ghost_per_site = [0 for i in range(len(ase_structure))]

    if len(magnetization_per_site) != len(ase_structure.numbers):
        raise ValueError(
            'The size of `magnetization_per_site` is different from the number of atoms.'
        )
    if len(ghost_per_site) != len(ase_structure.numbers):
        raise ValueError(
            'The size of `ghost_per_site` is different from the number of atoms.'
        )

    # Combine atom type with magnetizations and ghost_type
    complex_symbols = [
        f'{symbol}_{magn}_{ghost}'
        for symbol, magn, ghost in zip(ase_structure.get_chemical_symbols(),
                                       magnetization_per_site, ghost_per_site)
    ]

    # Assign a unique tag for every atom kind. Use OrderedDict for order
    unique_complex_symbols = list(
        collections.OrderedDict().fromkeys(complex_symbols).keys())
    combined = collections.OrderedDict()

    element_tag_counter = {}
    for c_symbol in unique_complex_symbols:
        element = c_symbol.split('_')[0]
        if element not in element_tag_counter:
            element_tag_counter[element] = 1
        else:
            element_tag_counter[element] += 1
        combined[c_symbol] = element_tag_counter[element]

    # Assigning correct tags to every atom.
    tags = [combined[key] for key in complex_symbols]
    ase_structure.set_tags(tags)

    kinds_dict = collections.OrderedDict()

    for c_symbol, tag in combined.items():
        element = c_symbol.split('_')[0]
        mag = float(c_symbol.split('_')[1])
        ghost = int(c_symbol.split('_')[2])

        kind_name = element + str(tag)
        info_dict = {'mag': mag, 'ghost': ghost}
        kinds_dict[kind_name] = info_dict

    return StructureData(ase=ase_structure), kinds_dict


def dict_merge(dct, merge_dct):
    """ Taken from https://gist.github.com/angstwad/bf22d1822c38a92ec0a9
    Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.
    :param dct: dict onto which the merge is executed
    :param merge_dct: dct merged into dct
    :return: None
    """
    for k in merge_dct.keys():
        if (k in dct and isinstance(dct[k], dict)
                and isinstance(merge_dct[k], collections.Mapping)):
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]


def get_cutoff(structure=None):
    if structure is None:
        return int(600)
    with open(pathlib.Path(__file__).parent / './data/atomic_kinds.yml',
              encoding='utf-8') as fhandle:
        atom_data = yaml.safe_load(fhandle)
    elements = structure.get_symbols_set()
    return max([atom_data['cutoff'][element] for element in elements])

def make_geom_file(atoms, filename, spin_guess=None):
        # spin_guess = [[spin_up_indexes], [spin_down_indexes]]
        tmpdir = tempfile.mkdtemp()
        file_path = tmpdir + "/" + filename

        orig_file = StringIO()
        atoms.write(orig_file, format='xyz')
        orig_file.seek(0)
        all_lines = orig_file.readlines()
        comment = all_lines[1] # with newline character!
        orig_lines = all_lines[2:]
        
        modif_lines = []
        for i_line, line in enumerate(orig_lines):
            new_line = line
            lsp = line.split()
            if spin_guess is not None:
                if i_line in spin_guess[0]:
                    new_line = lsp[0]+"1 " + " ".join(lsp[1:])+"\n"
                if i_line in spin_guess[1]:
                    new_line = lsp[0]+"2 " + " ".join(lsp[1:])+"\n"
            modif_lines.append(new_line)
        
        final_str = "%d\n%s" % (len(atoms), comment) + "".join(modif_lines)

        with open(file_path, 'w') as f:
            f.write(final_str)
        aiida_f = SinglefileData(file=file_path)
        shutil.rmtree(tmpdir)
        return aiida_f
