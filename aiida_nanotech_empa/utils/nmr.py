import ase
import ase.io
import ase.neighborlist
import ase.visualize
import numpy as np

from . import cycle_tools


def find_ref_points(ase_atoms_no_h, cycles, h=0.0):
    """Positive h means projection to z axis is positive and vice-versa."""

    centers, normals = cycle_tools.find_cycle_centers_and_normals(
        ase_atoms_no_h, cycles, h
    )

    ase_ref_p = ase.Atoms()

    for i_cyc in range(len(cycles)):
        if h == 0.0:
            ase_ref_p.append(ase.Atom("X", centers[i_cyc]))
        else:
            pos = centers[i_cyc] + np.abs(h) * normals[i_cyc]
            ase_ref_p.append(ase.Atom("X", pos))

    return ase_ref_p


def dist(p1, p2):
    if isinstance(p1, ase.Atom):
        p1 = p1.position
    if isinstance(p2, ase.Atom):
        p2 = p2.position
    return np.linalg.norm(p2 - p1)


def interp_pts(p1, p2, dx):
    vec = p2 - p1
    dist = np.sqrt(np.sum(vec**2))
    num_p = int(np.round(dist / dx))
    dx_real = dist / num_p
    dvec = vec / dist * dx_real

    points = np.outer(np.arange(0, num_p), dvec) + p1
    return points


def build_path(ref_pts, dx=0.1):

    point_arr = None

    for i_rp in range(len(ref_pts) - 1):

        pt1 = ref_pts[i_rp].position
        pt2 = ref_pts[i_rp + 1].position

        points = interp_pts(pt1, pt2, dx)

        if i_rp == len(ref_pts) - 2:
            points = np.concatenate([points, [pt2]], axis=0)

        if point_arr is None:
            point_arr = points
        else:
            point_arr = np.concatenate([point_arr, points], axis=0)

    ase_arr = ase.Atoms("X%d" % len(point_arr), point_arr)
    return ase_arr


def load_nics_gaussian(nics_path):

    sigma = []

    def extract_values(line):
        parts = line.split()
        return np.array([parts[1], parts[3], parts[5]], dtype=float)

    with open(nics_path) as file:
        lines = file.readlines()
        i_line = 0
        while i_line < len(lines):
            if "Bq   Isotropic" in lines[i_line]:
                s = np.zeros((3, 3))
                s[0] = extract_values(lines[i_line + 1])
                s[1] = extract_values(lines[i_line + 2])
                s[2] = extract_values(lines[i_line + 3])
                sigma.append(s)
                i_line += 4
            else:
                i_line += 1

    return np.array(sigma)


def is_number(x):
    try:
        float(x)
    except ValueError:
        return False
    else:
        return True


def parse_nmr_cmo_matrix(log_file_str, property_dict):

    lines = log_file_str.splitlines()

    # build the object
    n_atom = property_dict["natom"]
    n_occupied_mo = property_dict["homos"][0] + 1

    nmr_cmo_matrix = np.zeros((n_atom, n_occupied_mo, 3, 3))

    i_line = 0
    while i_line < len(lines):

        # --------------------------------------------------------------------------
        # Full Cartesian NMR shielding tensor (ppm) for atom  C(  1):
        # Canonical MO contributions
        #
        #   MO        XX      XY      XZ      YX      YY      YZ      ZX      ZY      ZZ
        # ===============================================================================
        #    1.      0.01    0.00    0.00    0.00    0.00    0.00    0.00    0.00   -0.16
        #    2.      0.01    0.00    0.00    0.00    0.00    0.00    0.00    0.00   -0.13
        #    3.      0.00    0.00    0.00    0.00    0.00    0.00    0.00    0.00    0.24
        #    4.      0.00    0.00    0.00    0.00   -0.03    0.00    0.00    0.00    0.27
        #    ...

        if "Full Cartesian NMR shielding tensor (ppm) for atom" in lines[i_line]:

            i_atom = int(lines[i_line].replace("(", ")").split(")")[-2]) - 1

            i_line += 1

            if "Canonical MO contributions" in lines[i_line]:

                for _i in range(2000):
                    i_line += 1
                    if "Total" in lines[i_line]:
                        break
                    split = lines[i_line].split()

                    if len(split) == 10 and is_number(split[-1]):
                        i_mo = int(split[0][:-1]) - 1

                        arr = np.array([float(x) for x in split[1:]])
                        nmr_cmo_matrix[i_atom, i_mo, :, :] = arr.reshape(3, 3)

        i_line += 1

    return nmr_cmo_matrix
