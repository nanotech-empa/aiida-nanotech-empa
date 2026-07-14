import ase

from aiida import orm, plugins
from aiida.manage import get_manager

Cp2kDiagWorkChain = plugins.WorkflowFactory("nanotech_empa.cp2k.diag")


def test_setup_fills_in_dft_params_defaults(aiida_profile, fixture_localhost):
    """Regression test: `setup` used to require `periodic`, `uks`, `elpa_switch`
    and `sc_diag` to be present in `dft_params`, raising a `KeyError` further
    down the outline if any of them were omitted. `setup` should now fill in
    defaults for these keys instead.
    """
    code = orm.InstalledCode(
        computer=fixture_localhost,
        filepath_executable="/bin/true",
        default_calc_job_plugin="cp2k",
    ).store()

    structure = orm.StructureData(
        ase=ase.Atoms(
            "H2",
            positions=[[0, 0, 0], [0, 0, 0.74]],
            cell=[10, 10, 10],
            pbc=True,
        )
    )

    runner = get_manager().get_runner()
    process = runner.instantiate_process(
        Cp2kDiagWorkChain,
        cp2k_code=code,
        structure=structure,
        dft_params=orm.Dict(dict={}),
    )

    process.setup()

    assert process.ctx.dft_params["periodic"] == "XYZ"
    assert process.ctx.dft_params["uks"] is False
    assert process.ctx.dft_params["elpa_switch"] is False
    assert process.ctx.dft_params["sc_diag"] is False
