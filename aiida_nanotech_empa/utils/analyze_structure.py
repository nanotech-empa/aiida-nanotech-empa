import numpy as np
import more_itertools as mit
from ase import Atoms
from ase import neighborlist
from scipy import sparse
import itertools
from scipy.signal import find_peaks
from scipy.spatial import ConvexHull
from copy import deepcopy

from traitlets import HasTraits, Instance, Dict, observe


def find_ranges(iterable):
    """Yield range of consecutive numbers."""
    for group in mit.consecutive_groups(iterable):
        group = list(group)
        if len(group) == 1:
            yield group[0]
        else:
            yield group[0], group[-1]


def list_to_string_range(lst, shift=1):
    """Converts a list like [0, 2, 3, 4] into a string like '1 3..5'.
    Shift used when e.g. for a user interface numbering starts from 1 not from 0"""
    return " ".join([
        f"{t[0] + shift}..{t[1] + shift}"
        if isinstance(t, tuple) else str(t + shift)
        for t in find_ranges(sorted(lst))
    ])


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


def conne_matrix(atoms):
    cutOff = neighborlist.natural_cutoffs(atoms)
    neighborList = neighborlist.NeighborList(cutOff,
                                             self_interaction=False,
                                             bothways=False)
    neighborList.update(atoms)

    return neighborList.get_connectivity_matrix()


def clusters(matrix):
    nclusters, idlist = sparse.csgraph.connected_components(matrix)
    return nclusters, [
        np.where(idlist == i)[0].tolist() for i in range(nclusters)
    ]


def molecules(ismol, atoms):
    if ismol:
        nmols, ids = clusters(conne_matrix(atoms[ismol]))
        return [[ismol[i] for i in ids[j]] for j in range(nmols)]
    return []


class StructureAnalyzer(HasTraits):
    structure = Instance(Atoms, allow_none=True)
    details = Dict()

    def gaussian(self, x, sig):
        return 1.0 / (sig * np.sqrt(2.0 * np.pi)) * np.exp(
            -np.power(x, 2.) / (2 * np.power(sig, 2.)))

    def boxfilter(self, x, thr):
        return np.asarray([1 if i < thr else 0 for i in x])

    ## Piero Gasparotto
    def get_types(self, frame):  # pylint: disable=(too-many-statements,too-many-locals) # noqa: C901
        # classify the atmos in:
        # 0=molecule
        # 1=slab atoms
        # 2=adatoms
        # 3=hydrogens on the surf
        # 5=unknown
        # 6=metalating atoms
        #frame=ase frame
        #thr=threashold in the histogram for being considered a surface layer
        nat = frame.get_global_number_of_atoms()

        #all atom types set to 5 (unknown)
        atype = np.zeros(nat, dtype=np.int16) + 5
        area = (frame.cell[0][0] * frame.cell[1][1])
        minz = np.min(frame.positions[:, 2])
        maxz = np.max(frame.positions[:, 2])

        if maxz - minz < 1.0:
            maxz += (1.0 - (maxz - minz)) / 2
            minz -= (1.0 - (maxz - minz)) / 2

        ##WHICH VALUES SHOULD WE USE BELOW??????
        sigma = 0.2  #thr
        peak_rel_height = 0.5
        layer_tol = 1.0 * sigma
        # quack estimate number atoms in a layer:
        nbins = int(np.ceil((maxz - minz) / 0.15))
        hist, _ = np.histogram(frame.positions[:, 2],
                               density=False,
                               bins=nbins)
        max_atoms_in_a_layer = max(hist)

        lbls = frame.get_chemical_symbols()
        n_intervals = int(np.ceil((maxz - minz + 3 * sigma) / (0.1 * sigma)))
        z_values = np.linspace(minz - 3 * sigma, maxz + 3 * sigma,
                               n_intervals)  #1000
        atoms_z_pos = frame.positions[:, 2]

        # OPTION 1: generate 2d array to apply the gaussian on
        z_v_exp, at_z_exp = np.meshgrid(z_values, atoms_z_pos)
        arr_2d = z_v_exp - at_z_exp
        atomic_density = np.sum(self.gaussian(arr_2d, sigma), axis=0)

        # OPTION 2: loop through atoms
        # atomic_density = np.zeros(z_values.shape)
        #for ia in range(len(atoms)):
        #    atomic_density += gaussian(z_values - atoms.positions[ia,2], sigma)

        peaks = find_peaks(atomic_density,
                           height=None,
                           threshold=None,
                           distance=None,
                           prominence=None,
                           width=None,
                           wlen=None,
                           rel_height=peak_rel_height)
        layersg = z_values[peaks[0].tolist()]

        ##check top and bottom layers should be documented better

        found_top_surf = False
        while not found_top_surf:
            iz = layersg[-1]
            twoD_atoms = [
                frame.positions[i, 0:2] for i in range(nat)
                if np.abs(frame.positions[i, 2] - iz) < layer_tol
            ]
            coverage = 0
            if len(twoD_atoms) > max_atoms_in_a_layer / 4:
                hull = ConvexHull(twoD_atoms)  ##
                coverage = hull.volume / area
            if coverage > 0.3:
                found_top_surf = True
            else:
                layersg = layersg[0:-1]

        found_bottom_surf = False
        while not found_bottom_surf:
            iz = layersg[0]
            twoD_atoms = [
                frame.positions[i, 0:2] for i in range(nat)
                if np.abs(frame.positions[i, 2] - iz) < layer_tol
            ]
            coverage = 0
            if len(twoD_atoms) > max_atoms_in_a_layer / 4:
                hull = ConvexHull(twoD_atoms)  ##
                coverage = hull.volume / area
            if coverage > 0.3 and len(twoD_atoms) > max_atoms_in_a_layer / 4:
                found_bottom_surf = True
            else:
                layersg = layersg[1:]

        bottom_z = layersg[0]
        top_z = layersg[-1]

        #check if there is a bottom layer of H
        found_layer_of_H = True
        for i in range(nat):
            iz = frame.positions[i, 2]
            if (layer_tol + iz) > bottom_z and iz < (bottom_z + layer_tol):
                if lbls[i] == 'H':
                    atype[i] = 3
                else:
                    found_layer_of_H = False
                    break
        if found_layer_of_H:
            layersg = layersg[1:]
            #bottom_z=layersg[0]

        layers_dist = []
        iprev = layersg[0]
        for inext in layersg[1:]:
            layers_dist.append(abs(iprev - inext))
            iprev = inext

        for i in range(nat):
            iz = frame.positions[i, 2]
            if (layer_tol + iz) > bottom_z and iz < (top_z + layer_tol):
                if not (atype[i] == 3 and found_layer_of_H):
                    atype[i] = 1
            else:
                if np.min([np.abs(iz - top_z),
                           np.abs(iz - bottom_z)]) < np.max(layers_dist):
                    if not (atype[i] == 3 and found_layer_of_H):
                        atype[i] = 2

        # assign the other types
        metalatingtypes = ('Au', 'Ag', 'Cu', 'Ni', 'Co', 'Zn', 'Mg')
        moltypes = ('H', 'N', 'B', 'O', 'C', 'F', 'S', 'Br', 'I', 'Cl')
        possible_mol_atoms = [
            i for i in range(nat) if atype[i] == 2 and lbls[i] in moltypes
        ]
        possible_mol_atoms += [i for i in range(nat) if atype[i] == 5]
        #identify separate molecules
        #all_molecules=self.molecules(mol_atoms,atoms)
        all_molecules = []
        if len(possible_mol_atoms) > 0:
            #conne = conne_matrix(frame[possible_mol_atoms])
            fragments = molecules(possible_mol_atoms, frame)
            all_molecules = deepcopy(fragments)
            #remove isolated atoms
            for frag in fragments:
                if len(frag) == 1:
                    all_molecules.remove(frag)
                else:
                    for atom in frag:
                        if lbls[atom] in metalatingtypes:
                            atype[atom] = 6
                        else:
                            atype[atom] = 0

        return atype, layersg, all_molecules

    def string_range_to_list(self, a):
        singles = [int(s) - 1 for s in a.split() if s.isdigit()]
        ranges = [r for r in a.split() if '..' in r]
        for r in ranges:
            t = r.split('..')
            to_add = [i - 1 for i in range(int(t[0]), int(t[1]) + 1)]
            singles += to_add
        return sorted(singles)

    @observe('structure')
    def _observe_structure(self, _=None):
        with self.hold_trait_notifications():
            self.details = self.analyze()

    def analyze(self):  # pylint: disable=(too-many-statements,too-many-locals) # noqa: C901
        if self.structure is None:
            return {}

        atoms = self.structure
        sys_size = np.ptp(atoms.positions, axis=0)
        no_cell = atoms.cell[0][0] < 0.1 or atoms.cell[1][
            1] < 0.1 or atoms.cell[2][2] < 0.1
        if no_cell:
            # set bounding box as cell
            atoms.cell = sys_size + 10

        atoms.set_pbc([True, True, True])

        total_charge = np.sum(atoms.get_atomic_numbers())
        bottom_H = []
        adatoms = []
        bulkatoms = []
        wireatoms = []
        metalatings = []
        unclassified = []
        slabatoms = []
        slab_layers = []
        all_molecules = None
        is_a_bulk = False
        is_a_molecule = False
        is_a_wire = False

        spins_up = [the_a.index for the_a in atoms if the_a.tag == 1]
        spins_down = [the_a.index for the_a in atoms if the_a.tag == 2]
        other_tags = [the_a.index for the_a in atoms if the_a.tag > 2]
        #### check if there is vacuum otherwise classify as bulk and skip

        vacuum_x = sys_size[0] + 4 < atoms.cell[0][0]
        vacuum_y = sys_size[1] + 4 < atoms.cell[1][1]
        vacuum_z = sys_size[2] + 4 < atoms.cell[2][2]
        # do not use a set in the following line list(set(atoms.get_chemical_symbols()))
        # need ALL atoms and elements for spin guess and for cost calculation
        all_elements = atoms.get_chemical_symbols()
        #cov_radii = [covalent_radii[a.number] for a in atoms]

        #nl = NeighborList(cov_radii, bothways = True, self_interaction = False)
        #nl.update(atoms)

        #metalating_atoms=['Ag','Au','Cu','Co','Ni','Fe']

        summary = ''
        cases = []
        if len(spins_up) > 0:
            summary += 'spins_up: ' + list_to_string_range(spins_up) + '\n'
        if len(spins_down) > 0:
            summary += 'spins_down: ' + list_to_string_range(spins_down) + '\n'
        if len(other_tags) > 0:
            summary += 'other_tags: ' + list_to_string_range(other_tags) + '\n'
        if (not vacuum_z) and (not vacuum_x) and (not vacuum_y):
            is_a_bulk = True
            sys_type = 'Bulk'
            cases = ['b']
            summary += 'Bulk contains: \n'
            slabatoms = list(range(len(atoms)))
            bulkatoms = slabatoms

        if vacuum_x and vacuum_y and vacuum_z:
            is_a_molecule = True
            sys_type = 'Molecule'
            summary += 'Molecule: \n'
            all_molecules = molecules(list(range(len(atoms))), atoms)
            com = np.average(atoms.positions, axis=0)
            summary += 'COM: ' + str(com) + ', min z: ' + str(
                np.min(atoms.positions[:, 2])) + '\n'
        if vacuum_x and vacuum_y and (not vacuum_z):
            is_a_wire = True
            sys_type = 'Wire'
            cases = ['w']
            summary += 'Wire along z contains: \n'
            slabatoms = list(range(len(atoms)))
        if vacuum_y and vacuum_z and (not vacuum_x):
            is_a_wire = True
            sys_type = 'Wire'
            cases = ['w']
            summary += 'Wire along x contains: \n'
            slabatoms = list(range(len(atoms)))
        if vacuum_x and vacuum_z and (not vacuum_y):
            is_a_wire = True
            sys_type = 'Wire'
            cases = ['w']
            summary += 'Wire along y contains: \n'
            slabatoms = list(range(len(atoms)))
            wireatoms = slabatoms
        ####END check
        is_a_slab = not (is_a_bulk or is_a_molecule or is_a_wire)

        if is_a_slab:
            cases = ['s']
            tipii, layersg, all_molecules = self.get_types(atoms)
            if vacuum_x:
                slabtype = 'YZ'
            elif vacuum_y:
                slabtype = 'XZ'
            else:
                slabtype = 'XY'

            sys_type = 'Slab' + slabtype
            mol_atoms = np.where(tipii == 0)[0].tolist()
            #mol_atoms=extract_mol_indexes_from_slab(atoms)
            metalatings = np.where(tipii == 6)[0].tolist()
            mol_atoms += metalatings

            ## bottom_H
            bottom_H = np.where(tipii == 3)[0].tolist()

            ## unclassified
            unclassified = np.where(tipii == 5)[0].tolist()

            slabatoms = np.where(tipii == 1)[0].tolist()
            adatoms = np.where(tipii == 2)[0].tolist()

            ##slab layers
            slab_layers = [[] for i in range(len(layersg))]
            for ia in slabatoms:
                idx = (np.abs(layersg - atoms.positions[ia, 2])).argmin()
                slab_layers[idx].append(ia)

            ##end slab layers
            summary += 'Slab ' + slabtype + ' contains: \n'
        summary += 'Cell: ' + " ".join(
            [str(i) for i in atoms.cell.diagonal().tolist()]) + '\n'
        if len(slabatoms) == 0:
            slab_elements = set([])
        else:
            slab_elements = set(atoms[slabatoms].get_chemical_symbols())

        if len(bottom_H) > 0:
            summary += 'bottom H: ' + list_to_string_range(bottom_H) + '\n'
        if len(slabatoms) > 0:
            summary += 'slab atoms: ' + list_to_string_range(slabatoms) + '\n'
        for nlayer, the_layer in enumerate(slab_layers):
            summary += 'slab layer ' + str(
                nlayer + 1) + ': ' + list_to_string_range(the_layer) + '\n'
        if len(adatoms) > 0:
            cases.append('a')
            summary += 'adatoms: ' + list_to_string_range(adatoms) + '\n'
        if all_molecules:
            cases.append('m')
            summary += '#' + str(len(all_molecules)) + ' molecules: '
            for nmols, the_mol in enumerate(all_molecules):
                summary += str(nmols +
                               1) + ') ' + list_to_string_range(the_mol)

        summary += ' \n'
        if len(metalatings) > 0:
            metalating_str = list_to_string_range(metalatings)
            summary += f'metal atoms inside molecules (already counted): {metalating_str}\n'
        if len(unclassified) > 0:
            cases.append('u')
            summary += 'unclassified: ' + list_to_string_range(unclassified)

        ## INDEXES FROM 0 if mol_ids_range is not called

        cell_str = " ".join(
            [str(i) for i in itertools.chain(*atoms.cell.tolist())])

        return {
            'total_charge': total_charge,
            'system_type': sys_type,
            'cell': cell_str,
            'slab_layers': slab_layers,
            'bottom_H': sorted(bottom_H),
            'bulkatoms': sorted(bulkatoms),
            'wireatoms': sorted(wireatoms),
            'slabatoms': sorted(slabatoms),
            'adatoms': sorted(adatoms),
            'all_molecules': all_molecules,
            'metalatings': sorted(metalatings),
            'unclassified': sorted(unclassified),
            'numatoms': len(atoms),
            'all_elements': all_elements,
            'slab_elements': slab_elements,
            'spins_up': spins_up,
            'spins_down': spins_down,
            'other_tags': other_tags,
            'sys_size': sys_size,
            'cases': cases,
            'summary': summary
        }
