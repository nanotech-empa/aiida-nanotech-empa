import pathlib

import ase.io
import click
from aiida import engine, orm, plugins

from aiida_nanotech_empa.utils.cube_utils import cube_from_qe_pp_arraydata

# AiiDA classes.
NanoribbonWorkChain = plugins.WorkflowFactory("nanotech_empa.nanoribbon")

DATA_DIR = pathlib.Path(__file__).parent.absolute()
OUTPUT_DIR = pathlib.Path(__file__).parent.absolute()


def _example_nanoribbon(
    opt_cell, qe_pw_code, qe_pp_code, qe_projwfc_code, geo_file, description
):
    builder = NanoribbonWorkChain.get_builder()

    # Calculation settings.
    builder.optimize_cell = orm.Bool(opt_cell)
    builder.max_kpoints = orm.Int(2)
    builder.precision = orm.Float(0.0)

    # Resources.
    builder.max_nodes = orm.Int(1)
    builder.mem_node = orm.Int(32)

    # Codes.
    builder.pw_code = qe_pw_code
    builder.pp_code = qe_pp_code
    builder.projwfc_code = qe_projwfc_code

    # Inputs
    builder.structure = orm.StructureData(ase=ase.io.read(geo_file))
    # builder.pseudo_family = orm.Str("SSSP_modified")
    builder.pseudo_family = orm.Str(
        "SSSP/1.2/PBE/efficiency"
    )  # It requires aiida-pseudo install sssp!

    # Metadata
    builder.metadata = {
        "description": description,
        "label": "NanoribbonWorkChain",
    }

    _, node = engine.run_get_node(builder)

    assert node.is_finished_ok

    if "spin_density_arraydata" in node.outputs:
        cube = cube_from_qe_pp_arraydata(node.outputs.spin_density_arraydata)
        cube.write_cube_file(OUTPUT_DIR / "spin.cube")
        print("Wrote spin.cube!")


def example_nanoribbon_no_spin(qe_pw_code, qe_pp_code, qe_projwfc_code):
    _example_nanoribbon(
        True,
        qe_pw_code,
        qe_pp_code,
        qe_projwfc_code,
        DATA_DIR / "c2h2_no_spin.xyz",
        "Test calculation no spin",
    )


def example_nanoribbon_spin(qe_pw_code, qe_pp_code, qe_projwfc_code):
    _example_nanoribbon(
        True,
        qe_pw_code,
        qe_pp_code,
        qe_projwfc_code,
        DATA_DIR / "c2h2_spin.xyz",
        "Test calculation spin",
    )


def example_nanoribbon_no_cell(qe_pw_code, qe_pp_code, qe_projwfc_code):
    _example_nanoribbon(
        False,
        qe_pw_code,
        qe_pp_code,
        qe_projwfc_code,
        DATA_DIR / "c2h2_no_spin.xyz",
        "Test calculation no cell opt",
    )


@click.command("cli")
@click.argument("pw_code", default="pw@localhost")
@click.argument("pp_code", default="pp@localhost")
@click.argument("projwfc_code", default="projwfc@localhost")
def run_all(pw_code, pp_code, projwfc_code):
    set_of_codes = (
        orm.load_code(pw_code),
        orm.load_code(pp_code),
        orm.load_code(projwfc_code),
    )
    example_nanoribbon_no_cell(*set_of_codes)
    example_nanoribbon_no_spin(*set_of_codes)
    example_nanoribbon_spin(*set_of_codes)


if __name__ == "__main__":
    run_all()
