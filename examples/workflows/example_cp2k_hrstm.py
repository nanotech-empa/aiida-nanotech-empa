import os
import pathlib

import ase.io
import click
import numpy as np
from aiida import engine, orm, plugins

Cp2kHrstmWorkChain = plugins.WorkflowFactory("nanotech_empa.cp2k.hrstm")

DATA_DIR = pathlib.Path(__file__).parent.absolute()
GEO_FILE = "c2h2_on_au111.xyz"


def _example_cp2k_hrstm(
    cp2k_code,
    afm_code,
    hrstm_code,
    sc_diag,
    force_multiplicity,
    uks,
    n_nodes,
    n_cores_per_node,
):
    # Check test geometry is already in database.
    qb = orm.QueryBuilder()
    qb.append(orm.Node, filters={"label": {"in": [GEO_FILE]}})
    structure = None
    for node_tuple in qb.iterall():
        node = node_tuple[0]
        structure = node
    if structure is not None:
        print(f"Found existing structure: {structure.pk}")
    else:
        structure = orm.StructureData(ase=ase.io.read(DATA_DIR / GEO_FILE))
        structure.label = GEO_FILE
        structure.store()
        print(f"Created new structure: {structure.pk}")
    builder = Cp2kHrstmWorkChain.get_builder()

    builder.metadata.label = "Cp2kHrstmWorkChain"
    builder.metadata.description = "test description"
    builder.cp2k_code = cp2k_code
    ase_geom = ase.io.read(DATA_DIR / GEO_FILE)
    builder.structure = orm.StructureData(ase=ase_geom)
    builder.options = {
        "resources": {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
        },
        "max_wallclock_seconds": 600,
    }
    builder.protocol = orm.Str("debug")

    if uks:
        dft_params = {
            "sc_diag": sc_diag,
            "force_multiplicity": force_multiplicity,
            "elpa_switch": False,
            "periodic": "XYZ",
            "uks": uks,
            "multiplicity": 1,
            "smear_t": 150,
            "spin_up_guess": [0],
            "spin_dw_guess": [1],
        }
    else:
        dft_params = {
            "sc_diag": sc_diag,
            "force_multiplicity": force_multiplicity,
            "elpa_switch": False,
            "periodic": "XYZ",
            "uks": uks,
            "smear_t": 150,
        }
    builder.dft_params = orm.Dict(dft_params)
    builder.ppm_code = afm_code

    cell = ase_geom.cell
    top_z = np.max(ase_geom.positions[:, 2])
    dx = 0.2
    scanminz = 3.5
    scanmaxz = 5.5
    amp = 1.4
    f0 = 22352.5

    ppm_params_dict = orm.Dict(
        {
            "Catom": 6,
            "Oatom": 8,
            "ChargeCuUp": -0.0669933,
            "ChargeCuDown": -0.0627402,
            "Ccharge": 0.212718,
            "Ocharge": -0.11767,
            "sigma": 0.7,
            "Cklat": 0.24600212465950813,
            "Oklat": 0.15085476515590224,
            "Ckrad": 20,
            "Okrad": 20,
            "rC0": [0.0, 0.0, 1.82806112489999961213],
            "rO0": [0.0, 0.0, 1.14881347770000097341],
            "PBC": "False",
            "gridA": list(cell[0]),
            "gridB": list(cell[1]),
            "gridC": list(cell[2]),
            "scanMin": [0.0, 0.0, np.round(top_z, 1) + scanminz],
            "scanMax": [cell[0, 0], cell[1, 1], np.round(top_z, 1) + scanmaxz],
            "scanStep": [dx, dx, dx],
            "Amplitude": amp,
            "f0Cantilever": f0,
            "tip": "None",
            "Omultipole": "s",
        }
    )
    builder.ppm_params = ppm_params_dict

    builder.hrstm_code = hrstm_code
    parent_dir = "parent_calc_folder/"
    ppm_dir = "ppm_calc_folder/"
    ppm_qk = ppm_dir + "Qo{:1.2f}Qc{:1.2f}K{:1.2f}/".format(
        ppm_params_dict["Ocharge"],
        ppm_params_dict["Ccharge"],
        ppm_params_dict["Oklat"],
    )
    path = os.path.dirname(hrstm_code.get_remote_exec_path()) + "/hrstm_tips/"
    pdos_list = [path + "tip_coeffs.tar.gz"]
    tip_pos = [ppm_qk + "PPpos", ppm_qk + "PPdisp"]

    hrstm_params = orm.Dict(
        {
            "--output": "hrstm",
            "--voltages": ["-0.3", "-0.1"],
            # Sample information
            "--cp2k_input_file": parent_dir + "aiida.inp",
            "--basis_set_file": parent_dir + "BASIS_MOLOPT",
            "--xyz_file": parent_dir + "aiida.coords.xyz",
            "--wfn_file": parent_dir + "aiida-RESTART.wfn",
            "--hartree_file": parent_dir + "aiida-HART-v_hartree-1_0.cube",
            "--emin": "-0.4",
            "--emax": "0",
            "--fwhm_sam": "0.1",
            "--dx_wfn": "0.2",
            "--extrap_dist": "4",
            "--wn": "5",
            # Tip information
            "--pdos_list": pdos_list,
            "--orbs_tip": "1",
            "--tip_shift": str(ppm_params_dict["rC0"][2] + ppm_params_dict["rO0"][2]),
            "--tip_pos_files": tip_pos,
            "--fwhm_tip": "0",
        }
    )
    builder.hrstm_params = hrstm_params

    _, calc_node = engine.run_get_node(builder)

    assert calc_node.is_finished_ok


def example_cp2k_hrstm_no_sc_diag(cp2k_code, afm_code, hrstm_code):
    _example_cp2k_hrstm(cp2k_code, afm_code, hrstm_code, False, True, False)


def example_cp2k_hrstm_sc_diag(cp2k_code, afm_code, hrstm_code):
    _example_cp2k_hrstm(cp2k_code, afm_code, hrstm_code, True, True, True)


@click.command("cli")
@click.argument("cp2k_code", default="cp2k@localhost")
@click.argument("ppafm2_code", default="2ppafm@localhost")
@click.argument("hrstm_code", default="hrstm@localhost")
@click.option("-n", "--n-nodes", default=1)
@click.option("-c", "--n-cores-per-node", default=1)
def run_all(cp2k_code, ppafm2_code, hrstm_code, n_nodes, n_cores_per_node):
    print("#### no sc_diag UKS no force")
    _example_cp2k_hrstm(
        orm.load_code(cp2k_code),
        orm.load_code(ppafm2_code),
        orm.load_code(hrstm_code),
        sc_diag=False,
        force_multiplicity=False,
        uks=True,
        n_nodes=n_nodes,
        n_cores_per_node=n_cores_per_node,
    )


if __name__ == "__main__":
    run_all()
