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


def compute_cost(element_list, calctype='default', uks=False):
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
        'Tb': 19,
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
    if calctype == 'slab':
        the_cost = int(the_cost / 11)
    else:
        the_cost = int(the_cost / 4)
    if uks:
        the_cost = the_cost * 1.26
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

    resources = {
        'slab': {
            50: {
                'nodes': 4,
                'tasks_per_node': max_tasks_per_node,
                'threads': 1
            },
            200: {
                'nodes': 12,
                'tasks_per_node': max_tasks_per_node,
                'threads': 1
            },
            1400: {
                'nodes': 27,
                'tasks_per_node': max_tasks_per_node,
                'threads': 1
            },
            3000: {
                'nodes': 48,
                'tasks_per_node': max_tasks_per_node,
                'threads': 1
            },
            4000: {
                'nodes': 75,
                'tasks_per_node': max_tasks_per_node,
                'threads': 1
            },
            10000: {
                'nodes': 108,
                'tasks_per_node': max_tasks_per_node,
                'threads': 1
            }
        },
        'default': {
            50: {
                'nodes': 4,
                'tasks_per_node': max_tasks_per_node,
                'threads': 1
            },
            100: {
                'nodes': 12,
                'tasks_per_node': max_tasks_per_node,
                'threads': 1
            },
            180: {
                'nodes': 27,
                'tasks_per_node': max_tasks_per_node,
                'threads': 1
            },
            400: {
                'nodes': 48,
                'tasks_per_node': max_tasks_per_node,
                'threads': 1
            },
        },
        'gw': {
            50: {
                'nodes': 12,
                'tasks_per_node': max_tasks_per_node,
                'threads': 1
            },
            100: {
                'nodes': 256,
                'tasks_per_node': int(max(max_tasks_per_node / 3, 1)),
                'threads': 1
            },
            180: {
                'nodes': 512,
                'tasks_per_node': max_tasks_per_node,
                'threads': 1
            },
            400: {
                'nodes': 1024,
                'tasks_per_node': max_tasks_per_node,
                'threads': 1
            },
        },
        'gw_ic': {
            50: {
                'nodes': 12,
                'tasks_per_node': max_tasks_per_node,
                'threads': 1
            },
            100: {
                'nodes': 256,
                'tasks_per_node': int(max(max_tasks_per_node / 3, 1)),
                'threads': 1
            },
            180: {
                'nodes': 512,
                'tasks_per_node': max_tasks_per_node,
                'threads': 1
            },
            400: {
                'nodes': 1024,
                'tasks_per_node': max_tasks_per_node,
                'threads': 1
            },
        }
    }

    cost = compute_cost(element_list=atoms.get_chemical_symbols(),
                        calctype=calctype,
                        uks=uks)

    theone = min(resources[calctype], key=lambda x: abs(x - cost))
    nodes = resources[calctype][theone]['nodes']
    tasks_per_node = resources[calctype][theone]['tasks_per_node']
    threads = resources[calctype][theone]['threads']
    return min(nodes, max_nodes), tasks_per_node, threads

def is_number(s):
    """ Returns True if string is a number or range. """
    numbers = s.split('..')
    try:
        return all(float(f) for f in numbers)
    except ValueError:
        return False
def fixed_dict(xyz,ids):
    return {'COMPONENTS_TO_FIX' : xyz,'LIST' : ids}
def collective_dict(cv,restraint):
    return {'RESTRAINT' : {'K' : restraint } , 'COLVAR': cv }
def get_atoms(details):
    """ Gets atom elements in a CV. """
    try:
        last_id = 1+min(i for i,j in enumerate(details[2:]) if not is_number(j) )
    except ValueError:
        last_id = len(details) - 1
    ids = ' '.join(i for i in details[2:last_id+1])
    return {'ATOMS':ids}

def get_ids(details,label=None):
    """ Get ids of atoms points and planes."""
    if label:
        labels = [i for i, x in enumerate(details) if x.lower() == label]
        ids = []
        for lab in labels:
            ids0=''
            pos = lab+2
            while is_number(details[pos]):
                ids0+=details[pos]+' '
                pos+=1
            ids.append(ids0)
    else:
        labels=[]
        ids = []
        pos = 1
        while is_number(details[pos]):
            ids.append(details[pos])
            pos+=1
    return labels,ids

def get_points(details):
    """ Get point elements in a CV. """
    points,allids = get_ids(details,'point')
    allpoints=[]
    for i,point in enumerate(points):
        ids = allids[i]
        if details[point+1].lower() == 'fix_point':
            allpoints.append({'TYPE':'FIX_POINT','XYZ':ids})
        else:
            allpoints.append({'TYPE':'GEO_CENTER','ATOMS':ids})
    return {'POINT':allpoints}

def get_planes(details):
    """ Get planes of a CV. """
    planes ,allids = get_ids(details,'plane')
    return_dict = {}
    allplanes=[]
    for i,plane in enumerate(planes):
        ids = allids[i]
        if details[plane+1].lower() == 'atoms':
            allplanes.append({'DEF_TYPE':'ATOMS','ATOMS':ids})
        else:
            allplanes.append({'DEF_TYPE':'VECTOR','NORMAL_VECTOR':ids})
    return {'PLANE':allplanes}

def cv_dist(details):
    """ CV for distance between atoms or points. """
    #WEIGHTS NOT IMPLEMENTED
    points = get_points(details)
    axis = [i for i, x in enumerate(details) if x.lower() == "axis"]
    return_dict={}
    # case ATOMS, no points
    if not points['POINT'] :
        return_dict = {'DISTANCE' : get_atoms(details)}
    # case of points
    else:
        return_dict={'DISTANCE':points}
    if axis:
        return_dict['DISTANCE'].update({'AXIS':details[axis[0]+1].upper()})
    return return_dict

def cv_angle(details):
    """ CV for angle between atoms. """
    #WEIGHTS NOT IMPLEMENTED
    points = get_points(details)
    axis = [i for i, x in enumerate(details) if x.lower() == "axis"]
    return_dict={}
    # case ATOMS, no points
    if not points['POINT'] :
        return_dict = {'ANGLE' : get_atoms(details)}
    # case of points
    else:
        return_dict={'ANGLE':points}
    return return_dict

def cv_angle_plane_plane(details):
    """ CV anagle between two planes. """
    points = get_points(details)
    return_dict = {'ANGLE_PLANE_PLANE':{}}
    if points:
        return_dict['ANGLE_PLANE_PLANE'].update(points)
    return_dict['ANGLE_PLANE_PLANE'].update(get_planes(details))
    return return_dict

def cv_bond_rotation(details):
    """ CV bond rotation. """
    points = get_points(details)
    if points['POINT']:
        return_dict={'BOND_ROTATION':points}
        return_dict['BOND_ROTATION'].update({'P1_BOND1': '1','P2_BOND1': '2','P1_BOND2': '3','P2_BOND2': '4'})
        return return_dict
    else:
        labels,ids = get_ids(details)
        return {'BOND_ROTATION':{'P1_BOND1': ids[0],'P2_BOND1': ids[1],'P1_BOND2': ids[2],'P2_BOND2': ids[3]}}

def get_colvar_section(colvars):
    """ Creates the COLVAR dictionary. """
        allcvs=[]
        colvars = colvars.split(",")
        for cv in colvars:
            details = cv.split()
            details.append('end')
            if details[0].lower() == 'distance':
                allcvs.append(cv_dist(details))
            elif details[0].lower() == 'angle':
                allcvs.append(cv_angle(details))
            elif details[0].lower() == 'angle_plane_plane':
                allcvs.append(cv_angle_plane_plane(details))
            elif details[0].lower() == 'bond_rotation':
                allcvs.append(cv_bond_rotation(details))
        return {'COLVAR':allcvs}

def get_constraints_section(constraints):
    """ Creates the constraints dictionary. """
    constraints_dict={}
    constraints = constraints.split(",")
    fixed=[]
    colvar=[]
    for const in constraints:
        details = const.split()
        details.append('end')
        if 'fixed' in details[0].lower():
            # 'fixed xy 1..2 7 9' --> '1..2 7 9'
            indexes = const.lower().replace('fixed','')
            indexes = indexes.replace('x','')
            indexes = indexes.replace('y','')
            indexes = indexes.replace('z','')
            xyz='XYZ'
            if any(c in details[1].upper() for c in ['X','Y','Z']):
                xyz = details[1].upper()
            fixed.append(fixed_dict(xyz,indexes.strip()))
        if 'collective' in details[0].lower():
            colvar.append(collective_dict(details[1],details[2]))
        if fixed:
            constraints_dict.update({'FIXED_ATOMS':fixed})
        if colvar:
            constraints_dict.update({'COLLECTIVE':colvar})
    return constraints_dict
