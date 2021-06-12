"""
Routines regarding gaussian cube files
"""

import numpy as np
import ase

import collections

from aiida_gaussian.utils.cube import Cube

ANG_TO_BOHR = 1.8897259886


def crop_cube(cube, x_crop=None, y_crop=None, z_crop=None):
    # pylint: disable=too-many-locals
    """
    Crops the extent of the cube file

    x_crop, y_crop, z_crop can be
        None: no cropping;
        single value: atomic extent gets added this value in negative and positive direction;
        tuple/list of two values: space kept in negative and positive directions;
    """

    # convert cell and origin to angstrom
    cell = cube.cell / ANG_TO_BOHR
    origin = cube.origin / ANG_TO_BOHR

    dv = np.diag(cell) / cube.data.shape

    # corners of initial box
    i_p0 = origin
    i_p1 = origin + np.diag(cell)

    # corners of cropped box
    c_p0 = np.copy(i_p0)
    c_p1 = np.copy(i_p1)

    for i, i_crop in enumerate([x_crop, y_crop, z_crop]):
        pmax = np.max(cube.ase_atoms.positions[:, i])
        pmin = np.min(cube.ase_atoms.positions[:, i])

        if i_crop:

            if isinstance(i_crop, collections.Iterable):
                i_crop_ = i_crop
            else:
                i_crop_ = [i_crop, i_crop]

            c_p0[i] = pmin - i_crop_[0]
            c_p1[i] = pmax + i_crop_[1]

            # make grids match
            diff_0 = np.round((c_p0[i] - i_p0[i]) / dv[i]) * dv[i]
            c_p0[i] = i_p0[i] + diff_0
            if c_p0[i] < i_p0[i]:
                c_p0[i] = i_p0[i]

            diff_1 = np.round((c_p1[i] - i_p1[i]) / dv[i]) * dv[i]
            c_p1[i] = i_p1[i] + diff_1
            if c_p1[i] > i_p1[i]:
                c_p1[i] = i_p1[i]

    # crop indexes
    crop_s = ((c_p0 - i_p0) / dv).astype(int)
    crop_e = cube.data.shape - ((i_p1 - c_p1) / dv).astype(int)

    # Update the cube
    cube.data = cube.data[crop_s[0]:crop_e[0], crop_s[1]:crop_e[1], crop_s[2]:
                          crop_e[2]]

    cube.origin = c_p0
    cube.cell = (np.atleast_2d(cube.data.shape).T * np.diag(dv)) * ANG_TO_BOHR
    cube.ase_atoms.positions = cube.ase_atoms.positions - cube.origin


def cube_from_qe_pp_arraydata(ad):

    data_units = str(ad.get_array('data_units'))
    coord_units = str(ad.get_array('coordinates_units'))
    data = ad.get_array('data')

    coords = ad.get_array('coordinates')
    dv = ad.get_array('voxel')

    # make coords and dv [au] if in [angstrom]
    if coord_units in ('angstrom', 'ang'):
        coords *= ANG_TO_BOHR
        dv *= ANG_TO_BOHR

    ase_atoms = ase.Atoms(ad.get_array('atomic_numbers'), coords / ANG_TO_BOHR)

    cell = dv * np.atleast_2d(data.shape).T

    return Cube(title="from arraydata node",
                comment=f"data units: {data_units}",
                ase_atoms=ase_atoms,
                cell=cell,
                data=data)
