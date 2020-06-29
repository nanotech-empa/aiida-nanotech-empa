"""Parsers provided by aiida_nanotech_empa.

Register parsers via the "aiida.parsers" entry point in setup.json.
"""
from .utils import clip_data, crop_cube, read_cube_file, write_cube_file
from aiida.plugins import ParserFactory

BasePpParser = ParserFactory('quantumespresso.pp')

class PpParser(BasePpParser):
    """Reduce and parse Gaussian Cube formatted output.
        :param data_file_str: the data file read in as a single string
    """
    def parse_gaussian(self, data_file_str):
        numbers, positions, cell, origin, data = read_cube_file(data_file_str.splitlines())
        new_data, new_cell, new_pos = crop_cube(data, positions, cell, origin, x_crop=None, y_crop=3.5, z_crop=3.5)
        clip_data(new_data, absmin=1e-4)
        cropped_data_file_str = write_cube_file(numbers, new_pos, new_cell, new_data)
        return super().parse_gaussian(cropped_data_file_str)

