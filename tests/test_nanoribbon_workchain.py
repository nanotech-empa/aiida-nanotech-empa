from types import SimpleNamespace

import ase

from aiida import orm, plugins
from aiida.common import AttributeDict
from aiida.manage import get_manager

from aiida_nanotech_empa.workflows.qe import nanoribbon

NanoribbonWorkChain = plugins.WorkflowFactory("nanotech_empa.nanoribbon")


def _code(computer, entry_point):
    return orm.InstalledCode(
        label=entry_point.rsplit(".", maxsplit=1)[-1],
        computer=computer,
        filepath_executable="/bin/true",
        default_calc_job_plugin=entry_point,
    ).store()


def test_projwfc_builder_is_compatible_with_aqe5(
    aiida_profile, fixture_localhost, monkeypatch
):
    """Test the AQE5 parameter casing and retrieval-list location."""
    structure = orm.StructureData(
        ase=ase.Atoms("C", positions=[[0, 0, 0]], cell=[8, 8, 8], pbc=True)
    )
    process = (
        get_manager()
        .get_runner()
        .instantiate_process(
            NanoribbonWorkChain,
            pw_code=_code(fixture_localhost, "quantumespresso.pw"),
            pp_code=_code(fixture_localhost, "quantumespresso.pp"),
            projwfc_code=_code(fixture_localhost, "quantumespresso.projwfc"),
            structure=structure,
            pseudo_family=orm.Str("unused"),
        )
    )

    process.ctx.nproc_mach = 1
    process.ctx.bands = SimpleNamespace(
        base=SimpleNamespace(
            attributes=SimpleNamespace(all={"resources": {"num_machines": 1}})
        ),
        inputs=AttributeDict(
            {
                "structure": structure,
                "parallelization": orm.Dict(dict={"npool": 1}),
            }
        ),
        outputs=AttributeDict(
            {
                "remote_folder": orm.RemoteData(
                    computer=fixture_localhost, remote_path="/tmp"
                ).store()
            }
        ),
    )

    submitted = {}

    def submit(_, builder):
        submitted["builder"] = builder
        return orm.CalcJobNode(computer=fixture_localhost).store()

    monkeypatch.setattr(nanoribbon.common_utils, "check_if_calc_ok", lambda *_: True)
    monkeypatch.setattr(NanoribbonWorkChain, "submit", submit)

    process.run_export_pdos()

    builder = submitted["builder"]
    assert builder.parameters.get_dict() == {
        "PROJWFC": {
            "ngauss": 1,
            "degauss": 0.007,
            "deltae": 0.01,
            "filproj": "projection.out",
        }
    }
    assert builder.settings.get_dict() == {"cmdline": ["-npools", "1"]}
    assert builder.metadata.options.additional_retrieve_list == [
        "./out/aiida.save/*.xml",
        "*_up",
        "*_down",
        "*_tot",
    ]
