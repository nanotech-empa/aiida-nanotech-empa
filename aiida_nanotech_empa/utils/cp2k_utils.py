# pylint: disable=too-many-locals
import pathlib
import yaml
import collections
import numpy as np
from aiida.orm import StructureData,Dict

ang_2_bohr = 1.889725989



def get_kinds_section(structure: StructureData, magnetization_tags=None):
    """ Write the &KIND sections given the structure and the settings_dict"""
    kinds = []
    with open(pathlib.Path(__file__).parent / '../files/cp2k/atomic_kinds.yml') as fhandle:
        atom_data = yaml.safe_load(fhandle)
    ase_structure = structure.get_ase()
    symbol_tag = {
        (symbol, str(tag)) for symbol, tag in zip(ase_structure.get_chemical_symbols(), ase_structure.get_tags())
    }
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

def get_kinds_section_gw(structure: StructureData, magnetization_tags=None,accuracy='lq'):
    """ Write the &KIND sections in gw calculations given the structure and the settings_dict"""
    bset = 'gw_basis_set'
    bsetaux = 'gw_basis_set_aux'
    if accuracy == 'hq':
        bset = 'gw_hq_basis_set'
        bsetaux = 'gw_hq_basis_set_aux'
    kinds = []
    with open(pathlib.Path(__file__).parent / '../files/cp2k/atomic_kinds.yml') as fhandle:
        atom_data = yaml.safe_load(fhandle)
    ase_structure = structure.get_ase()
    symbol_tag_mass = {
        (symbol, str(tag),str(int(mass))) for symbol, tag,mass in zip(ase_structure.get_chemical_symbols(), 
        ase_structure.get_tags(),ase_structure.get_masses())
    }
    for symbol, tag,mass in symbol_tag_mass:
        new_atom = {
            '_': symbol if tag == '0' else symbol + tag,
            'BASIS_SET': atom_data[bset][symbol],
            'BASIS_SET RI_AUX' : atom_data[bsetaux][symbol],
            'POTENTIAL': 'ALL',
        }
        if tag != '0':
            new_atom['ELEMENT'] = symbol
        if mass == '999':
            new_atom['GHOST'] = 'TRUE'
        if magnetization_tags:
            new_atom['MAGNETIZATION'] = magnetization_tags[tag]
        kinds.append(new_atom)
    return {'FORCE_EVAL': {'SUBSYS': {'KIND': kinds}}}


def tags_and_magnetization(structure, magnetization_per_site):
    """Gather the same atoms with the same magnetization into one atomic kind."""
    ase_structure = structure.get_ase()
    if magnetization_per_site:
        if len(magnetization_per_site) != len(ase_structure.numbers):
            raise ValueError('The size of `magnetization_per_site` is different from the number of atoms.')

        # Combine atom type with magnetizations.
        complex_symbols = [
            f'{symbol}_{magn}' for symbol, magn in zip(ase_structure.get_chemical_symbols(), magnetization_per_site)
        ]
        # Assign a unique tag for every atom kind. do not use set in enumerate to avoid random order...!
        combined = {symbol: tag + 1 for tag, symbol in enumerate(list(dict.fromkeys(complex_symbols).keys()))} 
        # Assigning correct tags to every atom.
        tags = [combined[key] for key in complex_symbols]
        ase_structure.set_tags(tags)
        # Tag-magnetization correspondance.
        tags_correspondance = {str(value): float(key.split('_')[1]) for key, value in combined.items()}
        return StructureData(ase=ase_structure), Dict(dict=tags_correspondance)
    # we force tags to be 0 if magnetization vector is not provided, this ensures we do not get a structure with unnecessary labels    
    else:
        tags = [0 for i in range(len(ase_structure))]
        return StructureData(ase=ase_structure), None

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
        if (k in dct and isinstance(dct[k], dict) and isinstance(merge_dct[k], collections.Mapping)):
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]

def get_cutoff(structure=None):
    cutoff = 600
    if structure is None:
        return int(600)
    with open(pathlib.Path(__file__).parent / '../files/cp2k/atomic_kinds.yml') as fhandle:
        atom_data = yaml.safe_load(fhandle)    
    elements=structure.get_symbols_set()
    return max([atom_data['cutoff'][element] for element in elements])


def compute_cost(element_list):
    cost={'H':1,'C':4,'Si':4,'N':5,'O':6,'Au':11,'Cu':11,'Ag':11,'Pt':18,'Co':11,'Zn':10,'Pd':18,'Ga':10}
    the_cost=0
    for element in element_list:
        s = ''.join(i for i in element if not i.isdigit())
        if isinstance(s[-1] ,type(1)):
            s=s[:-1]
        if s in cost.keys():
            the_cost+=cost[s]
        else:
            the_cost+=4
    return the_cost
            
def get_nodes(atoms=None,calctype='default',computer=None,max_nodes=1,uks=False):
    """"Determine the resources needed for th ecalculation."""
    threads = 1
    max_tasks_per_node = computer.get_default_mpiprocs_per_machine()
    if max_tasks_per_node is None:
        max_tasks_per_node=1
    GB_per_node = 32
    if atoms is None or computer is None:
        return max_nodes,max_tasks_per_node,threads
    
    GB_cell = 2.8*atoms.get_volume() / 2000
    cost=compute_cost(atoms.get_chemical_symbols())
    if uks:
        cost = cost*2
    nodes_cell = max(1,int(GB_cell / GB_per_node))
    if calctype == 'default':
        nodes = nodes_cell
        tasks = max_tasks_per_node

    elif calctype == 'slab':  
        tasks=max(max_tasks_per_node,int(324*cost/10000))
        list_squares=[i*max_tasks_per_node  for i in range(1,1200) 
                    if int(np.sqrt(i*max_tasks_per_node)) * int(np.sqrt(i*max_tasks_per_node)) == i*max_tasks_per_node ]
        takeClosest = lambda num,collection:min(collection,key=lambda x:abs(x-num))
        ideal = int(takeClosest(tasks,list_squares) / max_tasks_per_node )
        ratio = ideal*max_tasks_per_node / tasks
        nodes = ideal
        if ratio > 1.3 or ratio < 0.7:
            nodes = int(tasks/max_tasks_per_node)      
        tasks = max_tasks_per_node
    elif calctype == 'gw':
        use_tasks_per_node =  min(1,max_tasks_per_node)
        nodes = (max(int(cost/585),1))**3
        nodes = max(nodes,nodes_cell)
        tasks = nodes * use_tasks_per_node
        threads = min(2,max_tasks_per_node)
        use_tasks_per_node =  min(1,max_tasks_per_node)
        list_squares=[i*use_tasks_per_node  for i in range(1,4096) 
                    if int(np.sqrt(i*use_tasks_per_node)) * int(np.sqrt(i*use_tasks_per_node)) == i*use_tasks_per_node ]
        takeClosest = lambda num,collection:min(collection,key=lambda x:abs(x-num))
        nodes = int(takeClosest(tasks,list_squares) / use_tasks_per_node )   
        tasks = use_tasks_per_node
    elif calctype == 'gw_ic':
        use_tasks_per_node =  min(1,max_tasks_per_node)
        nodes = (max(int(cost/468),1))**3
        nodes = max(nodes,nodes_cell)
        tasks = nodes * use_tasks_per_node
        threads = min(2,max_tasks_per_node)
        use_tasks_per_node =  min(1,max_tasks_per_node)
        list_squares=[i*use_tasks_per_node  for i in range(1,4096) 
                    if int(np.sqrt(i*use_tasks_per_node)) * int(np.sqrt(i*use_tasks_per_node)) == i*use_tasks_per_node ]
        takeClosest = lambda num,collection:min(collection,key=lambda x:abs(x-num))
        nodes = int(takeClosest(tasks,list_squares) / use_tasks_per_node )    
        tasks = use_tasks_per_node    

    return min(nodes,max_nodes),tasks,threads
