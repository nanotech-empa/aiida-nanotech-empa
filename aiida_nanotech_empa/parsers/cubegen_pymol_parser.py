# -*- coding: utf-8 -*-
"""AiiDA-Gaussian output parser"""

import os
import tempfile

from aiida.orm import FolderData

import aiida_nanotech_empa.utils.pymol_render as pr

from aiida.plugins import ParserFactory

from aiida_gaussian.utils.cube import Cube

CubegenBaseParser = ParserFactory('gaussian.cubegen_base')


class CubegenPymolParser(CubegenBaseParser):
    """Cubegen parser based on pymol."""
    def _parse_folders(self, retrieved_folders, parser_params):

        # Parse constant-height planes based on the base parser
        super()._parse_folders(retrieved_folders, parser_params)

        # By default, don't re-orient cube
        orient_cube = False
        if 'orient_cube' in parser_params:
            orient_cube = parser_params['orient_cube']

        if 'isovalues' in parser_params:
            isovalues = parser_params['isovalues']
        else:
            isovalues = [0.01]

        with tempfile.TemporaryDirectory() as image_folder:

            for retrieved_fd in retrieved_folders:
                for filename in retrieved_fd.list_object_names():
                    if filename.endswith(".cube"):

                        # AiiDA retrieved folder provides handles to retrieved files
                        # pymol rendering, however, requires the path to the cube file
                        # therefore, we need to write the contents to a temporary file

                        with retrieved_fd.open(filename) as handle:
                            cube = Cube.from_file_handle(handle)

                        if orient_cube:
                            self._orient_cube(cube)

                        with tempfile.NamedTemporaryFile(mode='w+',
                                                         encoding='utf-8',
                                                         suffix='.cube') as tf:
                            cube.write_cube_file(tf.name)

                            for iv in isovalues:
                                pr.make_pymol_png(
                                    tf.name,
                                    isov=iv,
                                    colors=[
                                        (1.0, 0.2, 0.2),  # red
                                        (0.0, 0.4, 1.0),  # blue
                                    ],
                                    orientations=('z'),
                                    output_folder=image_folder.name,
                                    output_name=os.path.splitext(filename)[0])

            image_folder_node = FolderData(tree=image_folder.name)

            self.out("cube_image_folder", image_folder_node)
