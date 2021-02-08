"""Parsers provided by aiida_nanotech_empa.

Register parsers via the "aiida.parsers" entry point in setup.json.
"""
import numpy as np
from .utils import crop_cube, read_cube_file, write_cube_file

from aiida import orm
from aiida.plugins import ParserFactory

BasePpParser = ParserFactory('quantumespresso.pp')


class PpParser(BasePpParser):
    """Reduce and parse Gaussian Cube formatted output.
        :param data_file_str: the data file read in as a single string
    """
    def parse_gaussian(self, data_file_str):
        numbers, positions, cell, origin, data = read_cube_file(
            data_file_str.splitlines())
        new_data, new_cell, new_pos = crop_cube(data,
                                                positions,
                                                cell,
                                                origin,
                                                x_crop=None,
                                                y_crop=3.5,
                                                z_crop=6.2)
        # NB! No point in clipping if the file is not compressed!
        #clip_data(new_data, absmin=1e-8)
        cropped_data_file_str = write_cube_file(numbers, new_pos, new_cell,
                                                new_data)
        return self.parse_gaussian2(cropped_data_file_str)

    def parse_gaussian2(self, data_file_str):  # pylint: disable=too-many-locals
        """Parse Gaussian Cube formatted output.
        :param data_file_str: the data file read in as a single string
        """
        lines = data_file_str.splitlines()

        atoms_line = lines[2].split()
        natoms = int(atoms_line[0])  # The number of atoms listed in the file

        header = lines[:6 +
                       natoms]  # Header of the file: comments, the voxel, and the number of atoms and datapoints
        data_lines = lines[
            6 + natoms:]  # The actual data: atoms and volumetric data

        # Parse the declared dimensions of the volumetric data
        x_line = header[3].split()
        xdim = int(x_line[0])
        y_line = header[4].split()
        ydim = int(y_line[0])
        z_line = header[5].split()
        zdim = int(z_line[0])

        # Get the vectors describing the basis voxel
        voxel_array = np.array([[x_line[1], x_line[2], x_line[3]],
                                [y_line[1], y_line[2], y_line[3]],
                                [z_line[1], z_line[2], z_line[3]]],
                               dtype=np.float64)
        atomic_numbers = np.empty(natoms, int)
        coordinates = np.empty((natoms, 3))
        for i in range(natoms):
            line = header[6 + i].split()
            atomic_numbers[i] = int(line[0])
            coordinates[i] = [float(s) for s in line[2:]]

        # Get the volumetric data
        data_array = np.empty(xdim * ydim * zdim, dtype=float)
        cursor = 0
        for line in data_lines:
            ls = line.split()
            data_array[cursor:cursor + len(ls)] = ls
            cursor += len(ls)
        data_array = data_array.reshape((xdim, ydim, zdim))

        coordinates_units = 'bohr'
        data_units = self.units_dict[self.output_parameters['plot_num']]

        arraydata = orm.ArrayData()
        arraydata.set_array('voxel', voxel_array)
        arraydata.set_array('data', data_array)
        arraydata.set_array('data_units', np.array(data_units))
        arraydata.set_array('coordinates_units', np.array(coordinates_units))
        arraydata.set_array('coordinates', coordinates)
        arraydata.set_array('atomic_numbers', atomic_numbers)

        return arraydata
