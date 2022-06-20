import os
import os.path
import sys
import time
import tempfile

import numpy as np

import ase.io

from PIL import Image, ImageOps

from aiida_gaussian.utils.cube import Cube

# PYMOL asynchronous mode (pymolwiki.org/index.php/Jupyter)
import pymol  # pylint: disable=import-error,useless-suppression

_stdouterr = sys.stdout, sys.stderr
pymol.finish_launching(['pymol', '-qc'])
sys.stdout, sys.stderr = _stdouterr

# pylint: disable=import-error,useless-suppression,wrong-import-position
from pymol import cmd


def _crop_image_bbox(filename):

    image = Image.open(filename)

    try:
        image = image.convert('RGBa')
        image = image.crop(image.getbbox())
    except ValueError:
        inv_image = ImageOps.invert(image)
        image = image.crop(inv_image.getbbox())

    image = image.convert('RGBA')
    image.save(filename)


def _save_and_crop(fname, max_w, view):

    for _try in range(4):
        # Pymol sometimes creates the png with a delay
        # and sometimes even after a long delay, the image is not created
        # in this case, try to create the image again (loop over _try)
        cmd.png(fname, width=f"{max_w + 1}cm", dpi=500, ray=1)
        # Wait for a bit
        for _i in range(200):
            if os.path.isfile(fname):
                break
            time.sleep(0.5)
        if os.path.isfile(fname):
            break

    if not os.path.isfile(fname):
        raise FileNotFoundError(f'Pymol render did not create {fname}')

    cmd.set_view(view)  # reset the view
    # Sleep a bit more to make sure the image is finished
    # (before, sometimes the PIL in _crop was complaining of a corrupted png)
    time.sleep(0.1)
    _crop_image_bbox(fname)


def make_pymol_png(  # noqa
        input_file,
        isov=0.05,
        colors=('brightorange', 'marine'),
        output_folder='.',
        output_name=None,
        orientations=('z', 'y', 'x')):
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-statements

    cmd.delete('all')

    # ------------------------------------
    # General PYMOL settings

    # Transparency settings
    cmd.set(name="transparency_mode", value=1)
    cmd.set(name="backface_cull", value=0)
    cmd.set(name="two_sided_lighting", value=1)
    cmd.set(name="ray_shadows", value=0)
    cmd.set(name="depth_cue", value=0)

    # Ray tracing options (https://pymolwiki.org/index.php/Ray)
    cmd.set(name="ray_trace_mode", value=0)
    # outline
    #cmd.set(name="ray_trace_mode", value=1)
    # outline and "cartoony" appearance
    #cmd.set(name="ray_trace_mode", value=3)

    # ------------------------------------

    cmd.set("specular", 0.3)

    # ------------------------------------

    filepath, ext = os.path.splitext(input_file)
    filename = os.path.basename(filepath)

    if ext == '.cube':

        # Geometry from cube
        ase_geom = Cube.from_file(input_file, read_data=False).ase_atoms
        with tempfile.NamedTemporaryFile(delete=False, mode='w') as tempf:
            ase_geom.write(tempf.name, format='xyz')
            tempf.close()
            cmd.load(tempf.name, format='xyz')

        # Load the cube data & draw isosurfaces
        cmd.load(input_file, "cube")

        cmd.isosurface("pos_1", "cube", isov)
        cmd.set(name="surface_color", value=colors[0], selection="%pos_1")
        cmd.set(name="transparency", value=0.0, selection="%pos_1")

        cmd.isosurface("neg_1", "cube", -isov)
        cmd.set(name="surface_color", value=colors[1], selection="%neg_1")
        cmd.set(name="transparency", value=0.0, selection="%neg_1")

    elif ext == '.xyz':

        ase_geom = ase.io.read(input_file)
        cmd.load(input_file)

    x_w = np.ptp(ase_geom.positions[:, 0])
    y_w = np.ptp(ase_geom.positions[:, 1])
    z_w = np.ptp(ase_geom.positions[:, 2])

    max_w = np.max([x_w, y_w, z_w])

    # PYMOL geometry visualization

    cmd.hide("lines")
    cmd.show("sticks")
    cmd.show("spheres")

    cmd.set("stick_h_scale", 1.0)
    cmd.set_bond("stick_radius", 0.105, "all", 'all')

    cmd.set("sphere_scale", 0.21, "all")
    cmd.set("sphere_scale", 0.16, "elem H")

    cmd.color("gray10", "name C*")
    cmd.color("blue", "name N*")

    cmd.set("orthoscopic", "on")

    cmd.hide("spheres", "elem X")
    cmd.hide("sticks", "elem X")

    # ------------------------------------
    # Set the view
    # zoom to see molecule and extra z distance (1.5)
    cmd.zoom("(all)", buffer=1.5, complete=1)

    view = cmd.get_view()

    # ------------------------------------
    # Output images

    out_filepath = output_folder + "/"

    if output_name is None:
        out_filepath += filename
    else:
        out_filepath += output_name

    if ext == '.cube':
        out_filepath += "_iv{:.3f}".format(isov)

    for orient in orientations:
        if orient in ('z', 'z+'):
            _save_and_crop(out_filepath + "_z+.png", max_w, view)
        elif orient == 'z-':
            cmd.turn("x", 180)
            cmd.turn("y", 180)
            _save_and_crop(out_filepath + "_z-.png", max_w, view)
        elif orient in ('y', 'y+'):
            cmd.turn("x", -90)
            _save_and_crop(out_filepath + "_y+.png", max_w, view)
        elif orient == 'y-':
            cmd.turn("x", 90)
            cmd.turn("z", 180)
            _save_and_crop(out_filepath + "_y-.png", max_w, view)
        elif orient in ('x', 'x+'):
            cmd.turn("x", -90)
            cmd.turn("y", 90)
            _save_and_crop(out_filepath + "_x+.png", max_w, view)
        elif orient == 'x-':
            cmd.turn("x", -90)
            cmd.turn("y", -90)
            cmd.turn("z", 180)
            _save_and_crop(out_filepath + "_x-.png", max_w, view)
