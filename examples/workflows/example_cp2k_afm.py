import pathlib

import ase.io
import click
import numpy as np
from aiida import engine, orm, plugins

Cp2kAfmWorkChain = plugins.WorkflowFactory("nanotech_empa.cp2k.afm")

DATA_DIR = pathlib.Path(__file__).parent.absolute()
GEO_FILE = "c2h2_on_au111.xyz"


def _example_cp2k_afm(
    cp2k_code,
    afm_code1,
    sc_diag,
    force_multiplicity,
    uks,
    n_nodes,
    n_cores_per_node,
):
    structure = orm.StructureData(ase=ase.io.read(DATA_DIR / GEO_FILE))
    structure.store()
    builder = Cp2kAfmWorkChain.get_builder()

    builder.metadata.label = "CP2K_AFM"
    builder.metadata.description = "test description"
    builder.cp2k_code = cp2k_code
    ase_geom = ase.io.read(DATA_DIR / GEO_FILE)
    builder.structure = orm.StructureData(ase=ase_geom)
    builder.options = {
        "resources": {
            "num_machines": n_nodes,
            "num_mpiprocs_per_machine": n_cores_per_node,
        },
        "max_wallclock_seconds": 600,
    }

    builder.protocol = orm.Str("debug")
    builder.dft_params = orm.Dict(
        {
            "sc_diag": sc_diag,
            "force_multiplicity": force_multiplicity,
            "elpa_switch": False,
            "periodic": "XYZ",
            "uks": uks,
            "smear_t": 150,
        }
    )
    if uks:
        builder.dft_params = orm.Dict(
            {
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
        )

    builder.afm_pp_code = afm_code1

    cell = ase_geom.cell
    top_z = np.max(ase_geom.positions[:, 2])
    dx = 0.2
    scanminz = 3.5
    scanmaxz = 5.5
    amp = 1.4
    f0 = 22352.5
    builder.afm_pp_params = orm.Dict(
        {
            "probeType": "O",
            "charge": -0.028108681223969645,
            "sigma": 0.7,
            "tip": "s",
            "klat": 0.34901278868090491,
            "krad": 21.913190531846034,
            "r0Probe": [0.0, 0.0, 2.97],
            "PBC": "False",
            "gridA": list(cell[0]),
            "gridB": list(cell[1]),
            "gridC": list(cell[2]),
            "scanMin": [0.0, 0.0, np.round(top_z, 1) + scanminz],
            "scanMax": [cell[0, 0], cell[1, 1], np.round(top_z, 1) + scanmaxz],
            "scanStep": [dx, dx, dx],
            "Amplitude": amp,
            "f0Cantilever": f0,
        }
    )

    _, calc_node = engine.run_get_node(builder)

    assert calc_node.is_finished_ok


def example_cp2k_afm_no_sc_diag(cp2k_code, afm_code1):
    _example_cp2k_afm(cp2k_code, afm_code1, False, True, False)


def example_cp2k_afm_sc_diag(cp2k_code, afm_code1):
    _example_cp2k_afm(cp2k_code, afm_code1, True, True, True)


@click.command("cli")
@click.argument("cp2k_code", default="cp2k@localhost")
@click.argument("ppafm_code", default="ppafm@localhost")
@click.option("-n", "--n-nodes", default=1)
@click.option("-c", "--n-cores-per-node", default=1)
def run_all(cp2k_code, ppafm_code, n_nodes, n_cores_per_node):
    _example_cp2k_afm(
        cp2k_code=orm.load_code(cp2k_code),
        afm_code1=orm.load_code(ppafm_code),
        sc_diag=False,
        force_multiplicity=False,
        uks=False,
        n_nodes=n_nodes,
        n_cores_per_node=n_cores_per_node,
    )


if __name__ == "__main__":
    run_all()
