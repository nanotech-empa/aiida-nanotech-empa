from aiida import orm, plugins
from aiida.common.folders import SandboxFolder
from aiida.engine.utils import instantiate_process
from aiida.manage import get_manager

Cp2kUnfoldingCalculation = plugins.CalculationFactory("nanotech_empa.cp2k_unfolding")


def test_unfolding_additional_retrieve_list_extends(aiida_localhost):
    code = orm.InstalledCode(
        computer=aiida_localhost,
        filepath_executable="/bin/true",
        default_calc_job_plugin="nanotech_empa.cp2k_unfolding",
    ).store()
    process = instantiate_process(
        get_manager().get_runner(),
        Cp2kUnfoldingCalculation,
        code=code,
        parent_calc_folder=orm.RemoteData(
            computer=aiida_localhost, remote_path="/tmp/parent"
        ),
        primitive_vectors=orm.Str("1 0 0; 0 1 0"),
        settings=orm.Dict({"additional_retrieve_list": ["aiida.out"]}),
        metadata={"options": {"resources": {"num_machines": 1}}},
    )

    with SandboxFolder() as folder:
        calcinfo = process.prepare_for_submission(folder)

    assert calcinfo.retrieve_list == ["unfolding_bands.npz", "aiida.out"]
