"""Parsers provided by aiida_nanotech_empa.

Register parsers via the "aiida.parsers" entry point in setup.json.
"""
import numpy as np

from aiida_nanotech_empa.utils.cube_utils import crop_cube, read_cube_file

from aiida import orm
from aiida.plugins import ParserFactory

BasePpParser = ParserFactory('quantumespresso.pp')

ang_2_bohr = 1.889725989


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
                                                y_crop=1.8,
                                                z_crop=(3.1, 5.1))
        # NB! No point in clipping if the file is not compressed!
        #clip_data(new_data, absmin=1e-8)

        # Create the arraydata object
        coordinates_units = 'bohr'
        new_pos_au = new_pos * ang_2_bohr
        voxel_array = new_cell * ang_2_bohr / np.atleast_2d(new_data.shape).T

        data_units = self.units_dict[self.output_parameters['plot_num']]

        arraydata = orm.ArrayData()
        arraydata.set_array('voxel', voxel_array)
        arraydata.set_array('data', new_data)
        arraydata.set_array('data_units', np.array(data_units))
        arraydata.set_array('coordinates_units', np.array(coordinates_units))
        arraydata.set_array('coordinates', new_pos_au)
        arraydata.set_array('atomic_numbers', numbers)

        return arraydata
