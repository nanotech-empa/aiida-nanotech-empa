# -*- coding: utf-8 -*-
"""AiiDA-Gaussian output parser"""
from __future__ import absolute_import

import os
import tempfile

from aiida.orm import FolderData

import aiida_nanotech_empa.gaussian.utils.pymol_render as pr

from aiida_gaussian.parsers import CubegenBaseParser


class CubegenPymolParser(CubegenBaseParser):
    """Cubegen parser based on pymol"""
    def _parse_folders(self, retrieved_folder_paths, parser_params):

        if 'isovalues' in parser_params:
            isovalues = parser_params['isovalues']
        else:
            isovalues = [0.01]

        image_folder = tempfile.TemporaryDirectory()

        for folder_path in retrieved_folder_paths:
            for filename in os.listdir(folder_path):
                filepath = os.path.join(folder_path, filename)

                if filepath.endswith(".cube"):
                    for iv in isovalues:
                        pr.make_pymol_png(filepath,
                                          isov=iv,
                                          colors=['tv_red', (0.0, 0.4, 1.0)],
                                          output_folder=image_folder.name)

        image_folder_node = FolderData(tree=image_folder.name)

        self.out("cube_image_folder", image_folder_node)
