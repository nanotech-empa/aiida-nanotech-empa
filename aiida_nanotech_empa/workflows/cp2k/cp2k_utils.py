# pylint: disable=too-many-locals
import pathlib
import yaml
import collections
from aiida.orm import StructureData, Dict

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
        return StructureData(ase=ase_structure), Dict(dict=tags_correspondance)

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


def compute_cost(element_list):
    cost = {
        'H': 1,
        'C': 4,
        'Si': 4,
        'N': 5,
        'O': 6,
        'Au': 11,
        'Cu': 11,
        'Ag': 11,
        'Pt': 18,
        'Co': 11,
        'Zn': 10,
        'Pd': 18,
        'Ga': 10
    }
    the_cost = 0
    for element in element_list:
        s = ''.join(i for i in element if not i.isdigit())
        if isinstance(s[-1], type(1)):
            s = s[:-1]
        if s in cost.keys():
            the_cost += cost[s]
        else:
            the_cost += 4
    return the_cost


def get_nodes(atoms=None,
              calctype='default',
              computer=None,
              max_nodes=1,
              uks=False):
    """"Determine the resources needed for the calculation."""
    #pylint: disable=too-many-branches
    threads = 1
    max_tasks_per_node = computer.get_default_mpiprocs_per_machine()
    if max_tasks_per_node is None:
        max_tasks_per_node = 1

    if atoms is None or computer is None:
        return max_nodes, max_tasks_per_node, threads

    cost = compute_cost(atoms.get_chemical_symbols())
    if uks:
        cost = cost * 2

    if calctype == 'slab':
        if cost / 4 < 50:
            nodes = 4
            tasks_per_node = max_tasks_per_node
            threads = 1
        elif cost / 4 < 200:
            nodes = 12
            tasks_per_node = max_tasks_per_node
            threads = 1
        elif cost / 4 < 1200:
            nodes = 27
            tasks_per_node = max_tasks_per_node
            threads = 1
        else:
            nodes = 48
            tasks_per_node = max_tasks_per_node
            threads = 1

    elif calctype == 'default':
        if cost / 4 < 50:
            nodes = 4
            tasks_per_node = max_tasks_per_node
            threads = 1
        elif cost / 4 < 100:
            nodes = 12
            tasks_per_node = max_tasks_per_node
            threads = 1
        elif cost / 4 < 180:
            nodes = 12
            tasks_per_node = max_tasks_per_node
            threads = 1
        else:
            nodes = 27
            tasks_per_node = max_tasks_per_node
            threads = 1
    else:
        if cost / 4 < 50:
            nodes = 12
            tasks_per_node = max_tasks_per_node
            threads = 1
        elif cost / 4 < 100:
            nodes = 256
            tasks_per_node = int(max(max_tasks_per_node / 3, 1))
            threads = 1
        elif cost / 4 < 180:
            nodes = 512
            tasks_per_node = 1
            threads = 1
        else:
            nodes = 1024
            tasks_per_node = 1
            threads = 1

    return min(nodes, max_nodes), tasks_per_node, threads
