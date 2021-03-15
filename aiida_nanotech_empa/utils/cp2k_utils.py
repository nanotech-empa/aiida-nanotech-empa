# pylint: disable=too-many-locals
import pathlib
import yaml
import collections
from aiida.orm import StructureData, Dict

ang_2_bohr = 1.889725989


def get_kinds_section(structure: StructureData, magnetization_tags=None):
    """ Write the &KIND sections given the structure and the settings_dict"""
    kinds = []
    with open(
            pathlib.Path(__file__).parent /
            '../files/cp2k/atomic_kinds.yml') as fhandle:
        atom_data = yaml.safe_load(fhandle)
    ase_structure = structure.get_ase()
    symbol_tag = {(symbol, str(tag))
                  for symbol, tag in zip(ase_structure.get_chemical_symbols(),
                                         ase_structure.get_tags())}
    for symbol, tag in symbol_tag:
        new_atom = {
            '_': symbol if tag == '0' else symbol + tag,
            'BASIS_SET': atom_data['basis_set'][symbol],
            'POTENTIAL': atom_data['pseudopotential'][symbol],
        }
        if tag != '0':
            new_atom['ELEMENT'] = symbol
        if magnetization_tags:
            new_atom['MAGNETIZATION'] = magnetization_tags[tag]
        kinds.append(new_atom)
    return {'FORCE_EVAL': {'SUBSYS': {'KIND': kinds}}}


def get_kinds_section_gw(ase_structure=None, accuracy='lq'):
    """ Write the &KIND sections in gw calculations given the structure and the settings_dict"""
    bset = 'gw_basis_set'
    bsetaux = 'gw_basis_set_aux'
    potential = 'all'
    if accuracy == 'hq':
        bset = 'gw_hq_basis_set'
        bsetaux = 'gw_hq_basis_set_aux'
        potential = 'all'
    elif accuracy == 'st':
        bset = 'gw_st_basis_set'
        bsetaux = 'gw_st_basis_set_aux'
        potential = 'pseudopotential'
    kinds = []
    with open(
            pathlib.Path(__file__).parent /
            '../files/cp2k/atomic_kinds.yml') as fhandle:
        atom_data = yaml.safe_load(fhandle)

    for atom in ase_structure:
        new_atom = {
            '_':
            atom.symbol if atom.tag == '0' else atom.symbol + str(atom.tag),
            'BASIS_SET': atom_data[bset][atom.symbol],
            'BASIS_SET RI_AUX': atom_data[bsetaux][atom.symbol],
            'POTENTIAL': atom_data[potential][atom.symbol],
        }
        if atom.tag != '0':
            new_atom['ELEMENT'] = atom.symbol
        if atom.tag == 9999:
            new_atom['GHOST'] = 'TRUE'
        if atom.tag > 0 and atom.tag < 9999:
            new_atom['MAGNETIZATION'] = atom.tag
        kinds.append(new_atom)
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


def tags_and_magnetization_gw(structure, magnetization_per_site, ghosts):
    """Assigne to an atom a tag accordign to magnetization or ghost type. Magnetization = 2 means 1 imbalance in alpha/beta"""
    ase_structure = structure.get_ase()
    tags = [0 for a in ase_structure]
    if magnetization_per_site:
        tags = [int(m) for m in magnetization_per_site]
    if ghosts:
        for i in range(len(ghosts)):
            if ghosts[i]:
                tags[i] = 9999
    ase_structure.set_tags(tags)

    return StructureData(ase=ase_structure)


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
    with open(
            pathlib.Path(__file__).parent /
            '../files/cp2k/atomic_kinds.yml') as fhandle:
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
