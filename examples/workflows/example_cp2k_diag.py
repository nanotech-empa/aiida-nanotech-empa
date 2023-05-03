import pathlib

import ase.io
from aiida import engine, orm, plugins

Cp2kDiagWorkChain = plugins.WorkflowFactory("nanotech_empa.cp2k.diag")

DATA_DIR = pathlib.Path(__file__).parent.absolute()
GEO_FILE = "c2h2.xyz"


def _example_cp2k_diag(cp2k_code, sc_diag, force_multiplicity, uks):
    builder = Cp2kDiagWorkChain.get_builder()

    builder.metadata.label = "Cp2kDiagWorkChain"
    builder.metadata.description = "test description"
    builder.cp2k_code = cp2k_code
    ase_geom = ase.io.read(DATA_DIR / GEO_FILE)
    builder.structure = orm.StructureData(ase=ase_geom)
    builder.settings = orm.Dict(
        {
            "additional_retrieve_list": [
                "aiida.inp",
                "BASIS_MOLOPT",
                "aiida.coords.xyz",
                "aiida-RESTART.wfn",
            ]
        }
    )
    builder.dft_params = orm.Dict(
        {
            "protocol": "debug",
            "sc_diag": sc_diag,
            "force_multiplicity": force_multiplicity,
            "elpa_switch": False,
            "added_mos": 10,
            "uks": uks,
            "charge": 0,
            "periodic": "NONE",
            "smear_t": 150,
        }
    )
    if uks:
        builder.dft_params = orm.Dict(
            {
                "protocol": "debug",
                "sc_diag": sc_diag,
                "force_multiplicity": force_multiplicity,
                "elpa_switch": False,
                "added_mos": 10,
                "uks": uks,
                "charge": 0,
                "periodic": "NONE",
                "multiplicity": 1,
                "smear_t": 150,
                "spin_up_guess": [0],
                "spin_dw_guess": [1],
            }
        )
    builder.options = orm.Dict(
        {
            "max_wallclock_seconds": 600,
            "resources": {
                "num_machines": 1,
                "num_mpiprocs_per_machine": 1,
                "num_cores_per_mpiproc": 1,
            },
        }
    )

    _, calc_node = engine.run_get_node(builder)

    assert calc_node.is_finished_ok


def example_cp2k_diag_no_sc_diag(cp2k_code):
    _example_cp2k_diag(cp2k_code, False, True, False)


def example_cp2k_diag_sc_diag(cp2k_code):
    _example_cp2k_diag(cp2k_code, True, True, True)


if __name__ == "__main__":
    print("#### no sc_diag RKS")
    _example_cp2k_diag(orm.load_code("cp2k@localhost"), False, True, False)
    print("#### sc_diag RKS")
    _example_cp2k_diag(orm.load_code("cp2k@localhost"), True, True, False)
    print("#### no sc_diag UKS no force")
    _example_cp2k_diag(orm.load_code("cp2k@localhost"), False, False, True)
    print("#### sc_diag UKS force")
    _example_cp2k_diag(orm.load_code("cp2k@localhost"), True, True, True)
