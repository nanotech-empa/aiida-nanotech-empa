# pylint: disable=too-many-locals
import pathlib
import yaml
import numbers
import collections
import tempfile
import shutil
from io import StringIO
from aiida.orm import StructureData, Dict, SinglefileData, FolderData, load_node

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

def make_geom_file(atoms, filename, tags=None):
        tmpdir = tempfile.mkdtemp()

        singlefile = False
        if not isinstance(atoms,list):
            singlefile = True
            all_atoms=[atoms]
            all_filenames = [filename]
        for ifile, atoms in enumerate(all_atoms):
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
                if tags is not None :
                    if tags[i_line] == 0 :
                        new_line = lsp[0]+"  " + " ".join(lsp[1:])+"\n"
                    else:
                        new_line = lsp[0]+str(tags[i_line])+" " + " ".join(lsp[1:])+"\n"
                modif_lines.append(new_line)
            
            final_str = "%d\n%s" % (len(atoms), comment) + "".join(modif_lines)

            file_path = tmpdir + "/" + all_filenames[ifile]
            with open(file_path, 'w') as f:
                f.write(final_str)
        if singlefile:
            aiida_f = SinglefileData(file=file_path)            
        else:
            aiida_f = FolderData().replace_with_folder(folder=tmpdir)

        shutil.rmtree(tmpdir)    
        return aiida_f

# neb
def structure_available_wfn(node_uuid, relative_replica_id,current_hostname):
    """
    Checks availability of .wfn file corresponding to a structure and returns the remote path.
    """

    struct_node = load_node(node_uuid)

    if struct_node.creator is None:
        #print("Struct %d .wfn not avail: no creator." % struct_pk)
        return None

    parent_calc = struct_node.creator

    if parent_calc.computer is None:
        #print("Struct %d .wfn not avail: creator has no computer." % struct_pk)
        return None

    hostname = parent_calc.computer.hostname

    if hostname != current_hostname:
        #print("Struct %d .wfn not avail: different hostname." % struct_pk)
        return None

    if parent_calc.label == "neb":
        # it could be that the neb calculatio had a different number of replicas
        nreplica_parent = parent_calc.caller.inputs.neb_params['number_of_replica']
        ndigits = len(str(nreplica_parent))
        eff_replica_number = int(round(relative_replica_id*nreplica_parent,0)+1)
        # aiida-BAND2-RESTART.wfn 'replica_%s.xyz' % str(i +2 ).zfill(3)
        wfn_name = "aiida-BAND%s--RESTART.wfn" % str(eff_replica_number).zfill(ndigits)
    else:
        # In all other cases, e.g. geo opt, replica, ...
        # use the standard name
        wfn_name = "aiida-RESTART.wfn"

    wfn_search_path = parent_calc.get_remote_workdir() + "/" + wfn_name
    ssh_cmd = (
        "ssh "
        + hostname
        + " if [ -f "
        + wfn_search_path
        + " ]; then echo 1 ; else echo 0 ; fi"
    )
    wfn_exists = subprocess.check_output(ssh_cmd.split())

    if wfn_exists.decode()[0] != "1":
        #print("Struct %d .wfn not avail: file deleted from remote." % struct_pk)
        return None

    return wfn_search_path

def mk_wfn_cp_commands(nreplicas, replica_uuids, selected_computer):

    available_wfn_paths = []
    list_wfn_available = []
    list_of_cp_commands = []

    for ir, node_uuid in enumerate(replica_uuids):

        # in general teh number of uuids is <= nreplicas
        relative_replica_id = nreplicas*ir/len(replica_uuids)

        avail_wfn = structure_available_wfn(
            node_uuid,relative_replica_id, selected_computer.hostname
        )

        
        if avail_wfn:
            list_wfn_available.append(ir)  ## example:[0,4,8]
            available_wfn_paths.append(avail_wfn)

    if len(list_wfn_available) == 0:
        return []

    n_images_available = len(replica_uuids)
    n_images_needed = nreplicas
    n_digits = len(str(n_images_needed))
    fmt = "%." + str(n_digits) + "d"

    # assign each initial replica to a block of created reps
    block_size = n_images_needed / float(n_images_available)

    for to_be_created in range(1, n_images_needed + 1):
        name = "aiida-BAND" + str(fmt % to_be_created) + "-RESTART.wfn"

        lwa = np.array(list_wfn_available)

        # index_wfn = np.abs(np.round(lwa*block_size + block_size/2) - to_be_created).argmin()
        index_wfn = np.abs(
            lwa * block_size + block_size / 2 - to_be_created
        ).argmin()

        closest_available = lwa[index_wfn]

        #print(name, closest_available)

        list_of_cp_commands.append(f"cp {available_wfn_paths[index_wfn]} ./{name}")

    return list_of_cp_commands

# Constraints

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) * 180 / np.pi


def string_range_to_list(strng, shift=-1):
    """Converts a string like '1 3..5' into a list like [0, 2, 3, 4].
    Shift used when e.g. for a user interface numbering starts from 1 not from 0"""
    singles = [int(s) + shift for s in strng.split() if s.isdigit()]
    ranges = [r for r in strng.split() if ".." in r]
    if len(singles) + len(ranges) != len(strng.split()):
        return [], False
    for rng in ranges:
        try:
            start, end = rng.split("..")
            singles += [i + shift for i in range(int(start), int(end) + 1)]
        except ValueError:
            return [], False
    return singles, True


def is_number(s):
    """ Returns True if string is a number or a range. """
    num = s.split('..')
    try:
        return all(isinstance(float(f), numbers.Number) for f in num)
    except ValueError:
        return False


def fixed_dict(xyz, ids):
    """ Returns the CP2K input dictionary for fixed atoms."""
    return {'COMPONENTS_TO_FIX': xyz, 'LIST': ids}


def collective_dict(details):
    """ Returns the CP2K inpt dictionary for a restratint on a CV"""
    return {
        'RESTRAINT': {
            'K': details[1] + ' ' + details[2]
        },
        'COLVAR': details[0],
        'TARGET': details[3] + ' ' + details[4],
        'INTERMOLECULAR': ''
    }


def get_atoms(details):
    """ Gets atom elements in a stirng deifnition of a CP2K CV. """
    try:
        last_id = 1 + min(
            i for i, j in enumerate(details[2:]) if not is_number(j))
    except ValueError:
        last_id = len(details) - 1
    ids = ' '.join(i for i in details[2:last_id + 1])
    return {'ATOMS': ids}


def get_ids(details, label=None):
    """ Get ids of atoms points and planes in string definitions of CP2K CVs."""
    if label:
        labels = [i for i, x in enumerate(details) if x.lower() == label]
        ids = []
        for lab in labels:
            ids0 = ''
            pos = lab + 2
            while is_number(details[pos]):
                ids0 += details[pos] + ' '
                pos += 1
            ids.append(ids0)
    else:
        labels = []
        ids = []
        pos = 1
        while is_number(details[pos]):
            ids.append(details[pos])
            pos += 1
    return labels, ids


def get_points(details):
    """ Get point elements in a stirng deifnition of a CP2K CV. """
    points, allids = get_ids(details, 'point')
    allpoints = []
    for i, point in enumerate(points):
        ids = allids[i]
        if details[point + 1].lower() == 'fix_point':
            allpoints.append({'TYPE': 'FIX_POINT', 'XYZ': ids})
        else:
            allpoints.append({'TYPE': 'GEO_CENTER', 'ATOMS': ids})
    return {'POINT': allpoints}


# pylint: disable=unused-variable
def get_points_coords(points, atoms):
    """ Returns an ase Atoms object with H atoms positiones at the cartesian coordinates defined by the CP2K CV points."""
    coords = []
    for point in points:
        if point['TYPE'] == 'FIX_POINT':
            coords.append([float(i) for i in point['XYZ'].split()])
        else:
            ids, all_ok = string_range_to_list(point['ATOMS'], shift=-1)
            coords.append(
                np.mean(atoms.get_distances(ids[0], ids, mic=True,
                                            vector=True),
                        axis=0) + atoms[ids[0]].position)

    return Atoms(''.join(['H' for i in points]),
                 cell=atoms.cell,
                 positions=coords)


def get_planes(details):
    """ Returns CP2K input dictionary section for the planes of a  CV. """
    planes, allids = get_ids(details, 'plane')
    allplanes = []
    for i, plane in enumerate(planes):
        ids = allids[i]
        if details[plane + 1].lower() == 'atoms':
            allplanes.append({'DEF_TYPE': 'ATOMS', 'ATOMS': ids})
        else:
            allplanes.append({'DEF_TYPE': 'VECTOR', 'NORMAL_VECTOR': ids})

    return {'PLANE': allplanes}


def get_plane_normal(plane, atoms):
    """Computes the normal to a plane defined in  the CP2K input dict. MIC is used if the plane is defined by three atoms.
    The atoms input could be the result of a previous parsing of points"""
    if plane['DEF_TYPE'] == 'ATOMS':
        ids, all_ok = string_range_to_list(plane['ATOMS'], shift=-1)
        basis = atoms.get_distances(ids[1], [ids[0], ids[2]],
                                    mic=True,
                                    vector=True)
        normal = np.cross(basis[0], basis[1])
    else:
        normal = np.array([float(i) for i in plane['NORMAL_VECTOR'].split()])

    return normal


def get_planes_normals(planes, points, atoms):
    """ Returns the normal to the planes defined in the planes dictionaries of the CV angle_plane_plane."""

    if points:
        atoms_to_use = get_points_coords(points, atoms)
    else:
        atoms_to_use = atoms

    return [
        get_plane_normal(planes[0], atoms_to_use),
        get_plane_normal(planes[1], atoms_to_use)
    ]


def cv_dist(details):
    """ Returns the CP2K dictionary input section for the distance CV  between atoms or points. 
    cv_dist('distance point fix_point 1.1 2.2 3.3 point atoms 1..6   axis xy end'.split())
    returns
    {'DISTANCE': {'POINT': [{'TYPE': 'FIX_POINT', 'XYZ': '1.1 2.2 3.3 '},
    {'TYPE': 'GEO_CENTER', 'ATOMS': '1..6 '}], 'AXIS': 'XY'}}
    """
    #WEIGHTS NOT IMPLEMENTED
    points = get_points(details)
    axis = [i for i, x in enumerate(details) if x.lower() == "axis"]
    return_dict = {}
    # case ATOMS, no points
    if not points['POINT']:
        return_dict = {'DISTANCE': get_atoms(details)}

    # case of points
    else:
        return_dict = {'DISTANCE': points}

    if axis:
        return_dict['DISTANCE'].update({'AXIS': details[axis[0] + 1].upper()})

    return return_dict


def eval_cv_dist(details, atoms):
    """ Evaluates the CV distance between two atoms or points on the ase Atoms. """
    projections = {
        'X': [(1, 0, 0)],
        'Y': [(0, 1, 0)],
        'Z': [(0, 0, 1)],
        'XY': [(1, 0, 0), (0, 1, 0)],
        'XZ': [(1, 0, 0), (0, 0, 1)],
        'YZ': [(0, 1, 0), (0, 0, 1)],
        'XYZ': [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    }
    the_cv = cv_dist(details)
    if 'POINT' in the_cv['DISTANCE']:
        points_xyz = get_points_coords(the_cv['DISTANCE']['POINT'], atoms)
        dist = points_xyz.get_distance(0, 1, mic=True, vector=True)
    else:
        ids = [int(i) - 1 for i in the_cv['DISTANCE']['ATOMS'].split()
               ]  #indexes in ase start from 0
        dist = atoms.get_distance(ids[0], ids[1], mic=True, vector=True)
    axis = 'XYZ'
    if 'AXIS' in the_cv['DISTANCE']:
        axis = the_cv['DISTANCE']['AXIS']

    return [
        'distance',
        np.sqrt(
            np.sum([np.dot(dist, np.array(j))**2 for j in projections[axis]]))
    ]


def cv_angle(details):
    """ Returns the CP2K dictionary input section for the angle CV  between atoms or points. 
    cv_angle('angle point atoms 2 3  point fix_point 8.36 6.78 5.0 point atoms 3 4 end'.split())
    returns
    {'ANGLE': {'POINT': [{'TYPE': 'GEO_CENTER', 'ATOMS': '2 3 '}, {'TYPE': 'FIX_POINT', 'XYZ': '8.36 6.78 5.0 '},
    {'TYPE': 'GEO_CENTER', 'ATOMS': '3 4 '}], 'ATOMS': '1 2 3'}}
    """
    #WEIGHTS NOT IMPLEMENTED
    points = get_points(details)
    return_dict = {}
    # case ATOMS, no points
    if not points['POINT']:
        return_dict = {'ANGLE': get_atoms(details)}

    # case of points
    else:
        return_dict = {'ANGLE': points}
        return_dict['ANGLE'].update({'ATOMS': '1 2 3'})

    return return_dict


def eval_cv_angle(details, atoms):
    """ Evaluates the CV angle between the atoms defined in the CV. """
    the_cv = cv_angle(details)
    if 'POINT' not in the_cv['ANGLE']:
        ids = [int(i) - 1 for i in the_cv['ANGLE']['ATOMS'].split()
               ]  # shift -1 atom indexes
        return ['angle', atoms.get_angle(ids[0], ids[1], ids[2], mic=True)]
    #else:
    points_xyz = get_points_coords(the_cv['ANGLE']['POINT'], atoms)
    return ['angle', points_xyz.get_angle(0, 1, 2, mic=True)]


def cv_angle_plane_plane(details):
    """ CV anagle between two planes. 
    cv_angle_plane_plane('angle_plane_plane point fix_point 12.1 7.5  5.  
    point atoms 1..6 point fix_point 7.1 7.5   7. plane atoms 1 2 3 plane vector 0 0 1 end'.split())
    returns
    {'ANGLE_PLANE_PLANE': {'POINT': [{'TYPE': 'FIX_POINT', 'XYZ': '12.1 7.5 5. '},
    {'TYPE': 'GEO_CENTER', 'ATOMS': '1..6 '},
    {'TYPE': 'FIX_POINT', 'XYZ': '7.1 7.5 7. '}],
    'PLANE': [{'DEF_TYPE': 'ATOMS', 'ATOMS': '1 2 3 '},
    {'DEF_TYPE': 'VECTOR', 'NORMAL_VECTOR': '0 0 1 '}]}}
    """
    points = get_points(details)
    return_dict = {'ANGLE_PLANE_PLANE': {}}
    if points:
        return_dict['ANGLE_PLANE_PLANE'].update(points)
    return_dict['ANGLE_PLANE_PLANE'].update(get_planes(details))
    return return_dict


def eval_cv_angle_plane_plane(details, atoms):
    """ Evaluate the CV angle between two planes."""
    the_cv = cv_angle_plane_plane(details)
    points = []
    if 'POINT' in the_cv['ANGLE_PLANE_PLANE']:
        points = the_cv['ANGLE_PLANE_PLANE']['POINT']
    normals = get_planes_normals(the_cv['ANGLE_PLANE_PLANE']['PLANE'], points,
                                 atoms)
    return ['angle', angle_between(normals[0], normals[1])]


def cv_bond_rotation(details):
    """ Function to compute the CP2K input dictionary for the CV bond rotation.
    
    cv_bond_rotation('bond_rotation point fix_point 8.36 6.78 5.8 point fix_point 0 0 1 point atoms 3 point atoms 9 end'.split())
    
    returns
    
   {'BOND_ROTATION': {'POINT': [{'TYPE': 'FIX_POINT', 'XYZ': '8.36 6.78 5.8 '},
   {'TYPE': 'FIX_POINT', 'XYZ': '0 0 1 '},
   {'TYPE': 'GEO_CENTER', 'ATOMS': '3 '},
   {'TYPE': 'GEO_CENTER', 'ATOMS': '9 '}],
   'P1_BOND1': '1',
   'P2_BOND1': '2',
   'P1_BOND2': '3',
   'P2_BOND2': '4'}}
    """
    points = get_points(details)
    if points['POINT']:
        return_dict = {'BOND_ROTATION': points}
        return_dict['BOND_ROTATION'].update({
            'P1_BOND1': '1',
            'P2_BOND1': '2',
            'P1_BOND2': '3',
            'P2_BOND2': '4'
        })
        return return_dict
    #else:
    labels, ids = get_ids(details)
    return {
        'BOND_ROTATION': {
            'P1_BOND1': ids[0],
            'P2_BOND1': ids[1],
            'P1_BOND2': ids[2],
            'P2_BOND2': ids[3]
        }
    }


def eval_cv_bond_rotation(details, atoms):
    """ Function to evaluate the CV bond rotation."""
    the_cv = cv_bond_rotation(details)
    if 'POINT' not in the_cv['BOND_ROTATION']:
        ids = [
            int(the_cv['BOND_ROTATION']['P1_BOND1']),
            int(the_cv['BOND_ROTATION']['P2_BOND1']),
            int(the_cv['BOND_ROTATION']['P1_BOND2']),
            int(the_cv['BOND_ROTATION']['P2_BOND2'])
        ]
        return [
            'angle',
            angle_between(
                atoms.get_distance(ids[0], ids[1], mic=True, vector=True),
                atoms.get_distance(ids[2], ids[3], mic=True, vector=True))
        ]
    #else:
    points_xyz = get_points_coords(the_cv['BOND_ROTATION']['POINT'], atoms)
    return [
        'angle',
        angle_between(points_xyz.get_distance(0, 1, mic=True, vector=True),
                      points_xyz.get_distance(2, 3, mic=True, vector=True))
    ]


def get_colvars_section(colvars):
    """ Creates the COLVAR CP2K input dictionary. """
    allcvs = []
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

    return {'COLVAR': allcvs}


def get_constraints_section(constraints):
    """ Creates the CONSTRAINTS CP2K input dictionary. """
    constraints_dict = {}
    constraints = constraints.split(",")
    fixed = []
    colvar = []
    for const in constraints:
        details = const.split()
        details.append('end')
        if 'fixed' in details[0].lower():
            # 'fixed xy 1..2 7 9' --> '1..2 7 9'
            indexes = const.lower().replace('fixed', '')
            indexes = indexes.replace('x', '')
            indexes = indexes.replace('y', '')
            indexes = indexes.replace('z', '')
            xyz = 'XYZ'
            if any(c in details[1].upper() for c in ['X', 'Y', 'Z']):
                xyz = details[1].upper()
            fixed.append(fixed_dict(xyz, indexes.strip()))
        if 'collective' in details[0].lower():
            colvar.append(collective_dict(details[1:]))
        if fixed:
            constraints_dict.update({'FIXED_ATOMS': fixed})
        if colvar:
            constraints_dict.update({'COLLECTIVE': colvar})

    return constraints_dict


def compute_colvars(colvars, atoms):
    """ Computes the values of the colvars from a list of colvars and ase Atoms. """
    allcvs = []
    colvars = colvars.split(",")
    for cv in colvars:
        details = cv.split()
        details.append('end')
        if details[0].lower() == 'distance':
            allcvs.append(eval_cv_dist(details, atoms))
        elif details[0].lower() == 'angle':
            allcvs.append(eval_cv_angle(details, atoms))
        elif details[0].lower() == 'angle_plane_plane':
            allcvs.append(eval_cv_angle_plane_plane(details, atoms))
        elif details[0].lower() == 'bond_rotation':
            allcvs.append(eval_cv_bond_rotation(details, atoms))

    return allcvs