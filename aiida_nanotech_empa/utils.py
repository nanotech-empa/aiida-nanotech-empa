# pylint: disable=too-many-locals
import numpy as np

ang_2_bohr = 1.889725989


def read_cube_file(file_lines):

    fit = iter(file_lines)

    # Title and comment:
    _ = next(fit)  # noqa
    _ = next(fit)  # noqa

    line = next(fit).split()
    natoms = int(line[0])

    origin = np.array(line[1:], dtype=float)

    shape = np.empty(3, dtype=int)
    cell = np.empty((3, 3))
    for i in range(3):
        n, x, y, z = [float(s) for s in next(fit).split()]
        shape[i] = int(n)
        cell[i] = n * np.array([x, y, z])

    numbers = np.empty(natoms, int)
    positions = np.empty((natoms, 3))
    for i in range(natoms):
        line = next(fit).split()
        numbers[i] = int(line[0])
        positions[i] = [float(s) for s in line[2:]]

    positions /= ang_2_bohr  # convert from bohr to ang

    data = np.empty(shape[0] * shape[1] * shape[2], dtype=float)
    cursor = 0
    for i, line in enumerate(fit):
        ls = line.split()
        data[cursor:cursor + len(ls)] = ls
        cursor += len(ls)

    data = data.reshape(shape)

    cell /= ang_2_bohr  # convert from bohr to ang

    return numbers, positions, cell, origin, data


def clip_data(data, absmin=None, absmax=None):
    if absmin:
        data[np.abs(data) < absmin] = 0
    if absmax:
        data[data > absmax] = absmax
        data[data < -absmax] = -absmax  # pylint: disable=invalid-unary-operand-type


def crop_cube(data, pos, cell, origin, x_crop=None, y_crop=None, z_crop=None):  # pylint: disable=too-many-arguments

    dv = np.diag(cell) / data.shape

    # corners of initial box
    i_p0 = origin
    i_p1 = origin + np.diag(cell)

    # corners of cropped box
    c_p0 = np.copy(i_p0)
    c_p1 = np.copy(i_p1)

    for i, i_crop in enumerate([x_crop, y_crop, z_crop]):
        pmax, pmin = np.max(pos[:, i]), np.min(pos[:, i])

        if i_crop:
            c_p0[i] = pmin - i_crop / 2
            c_p1[i] = pmax + i_crop / 2

            # make grids match
            shift_0 = (c_p0[i] - i_p0[i]) % dv[i]
            c_p0[i] -= shift_0

            shift_1 = (c_p1[i] - i_p0[i]) % dv[i]
            c_p1[i] -= shift_1

    # crop
    crop_s = ((c_p0 - i_p0) / dv).astype(int)
    crop_e = data.shape - ((i_p1 - c_p1) / dv).astype(int)

    data = data[crop_s[0]:crop_e[0], crop_s[1]:crop_e[1], crop_s[2]:crop_e[2]]

    origin = c_p0

    new_cell = c_p1 - c_p0

    # make new origin 0,0,0
    new_pos = pos - origin

    return data, np.diag(new_cell), new_pos


def write_cube_file(numbers,
                    positions,
                    cell,
                    data,
                    origin=np.array([0.0, 0.0, 0.0])):
    filecontent = "Clipped-cropped cube file.\n"

    positions *= ang_2_bohr
    origin *= ang_2_bohr

    natoms = positions.shape[0]

    filecontent += 'cube\n'

    dv_br = cell * ang_2_bohr / data.shape

    filecontent += "{:5d} {:12.6f} {:12.6f} {:12.6f}\n".format(
        natoms, origin[0], origin[1], origin[2])

    for i in range(3):
        filecontent += "{:5d} {:12.6f} {:12.6f} {:12.6f}\n".format(
            data.shape[i], dv_br[i][0], dv_br[i][1], dv_br[i][2])

    for i in range(natoms):
        at_x, at_y, at_z = positions[i]
        filecontent += "{:5d} {:12.6f} {:12.6f} {:12.6f} {:12.6f}\n".format(
            numbers[i], 0.0, at_x, at_y, at_z)

    fmt = ' {:11.4e}'
    for ix in range(data.shape[0]):
        for iy in range(data.shape[1]):
            for line in range(data.shape[2] // 6):
                filecontent += (fmt * 6 + "\n").format(*data[ix, iy, line *
                                                             6:(line + 1) * 6])
            left = data.shape[2] % 6
            if left != 0:
                filecontent += (fmt * left +
                                "\n").format(*data[ix, iy, -left:])

    return filecontent
