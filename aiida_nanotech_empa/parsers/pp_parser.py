import io

import numpy as np
from aiida import orm
from aiida_gaussian.utils.cube import Cube
from aiida_quantumespresso.parsers.pp import PpParser as BasePpParser

from aiida_nanotech_empa.utils.cube_utils import crop_cube

from ..helpers import ANG_TO_BOHR


class PpParser(BasePpParser):
    """Reduce and parse Gaussian Cube formatted output.
    :param data_file_str: the data file read in as a single string
    """

    @staticmethod
    def parse_gaussian(data_file_str, data_units):
        with io.StringIO(data_file_str) as data_file_handle:
            cube = Cube.from_file_handle(data_file_handle)

        crop_cube(cube, x_crop=None, y_crop=1.8, z_crop=3.1)

        # Create the arraydata object
        coordinates_units = "bohr"
        new_pos_au = cube.ase_atoms.positions * ANG_TO_BOHR
        voxel_array = cube.cell / np.atleast_2d(cube.data.shape).T
        numbers = cube.ase_atoms.numbers

        arraydata = orm.ArrayData()
        arraydata.set_array("voxel", voxel_array)
        arraydata.set_array("data", cube.data)
        arraydata.set_array("data_units", np.array(data_units))
        arraydata.set_array("coordinates_units", np.array(coordinates_units))
        arraydata.set_array("coordinates", new_pos_au)
        arraydata.set_array("atomic_numbers", numbers)

        return arraydata
