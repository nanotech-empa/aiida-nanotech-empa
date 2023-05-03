from aiida.engine import run_get_node
from aiida.orm import Bool, Float, Int, List, Str, StructureData, load_code
from aiida.plugins import WorkflowFactory
from ase import Atoms

Cp2kMoleculeGwWorkChain = WorkflowFactory("nanotech_empa.cp2k.molecule_gw")


def _example_cp2k_gw(cp2k_code, ic, protocol, mult):
    builder = Cp2kMoleculeGwWorkChain.get_builder()

    builder.metadata.label = "Cp2kMoleculeGwWorkChain"
    builder.metadata.description = "test description"
    builder.code = cp2k_code

    ase_geom = Atoms("HH", positions=[[0, 0, 0], [0.75, 0, 0]], cell=[4.0, 4.0, 4.0])
    builder.structure = StructureData(ase=ase_geom)

    builder.protocol = Str(protocol)

    builder.multiplicity = Int(mult)
    if mult == 1:
        builder.magnetization_per_site = List([-1, 1])

    builder.run_image_charge = Bool(ic)
    builder.z_ic_plane = Float(0.8)

    builder.debug = Bool(True)
    builder.options.scf = {
        "max_wallclock_seconds": 600,
        "resources": {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
            "num_cores_per_mpiproc": 1,
        },
    }

    builder.options.gw = {
        "max_wallclock_seconds": 600,
        "resources": {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
            "num_cores_per_mpiproc": 1,
        },
    }

    _, calc_node = run_get_node(builder)

    assert calc_node.is_finished_ok

    gw_out_dict = dict(calc_node.outputs.gw_output_parameters)
    print()
    for k in gw_out_dict:
        print(f"  {k}: {gw_out_dict[k]}")


def example_cp2k_gw_gpw_std_rks(cp2k_code):
    _example_cp2k_gw(cp2k_code, False, "gpw_std", 0)


def example_cp2k_gw_gpw_std_uks(cp2k_code):
    _example_cp2k_gw(cp2k_code, False, "gpw_std", 1)


def example_cp2k_gw_gapw_std_rks(cp2k_code):
    _example_cp2k_gw(cp2k_code, False, "gapw_std", 0)


def example_cp2k_gw_gapw_std_uks(cp2k_code):
    _example_cp2k_gw(cp2k_code, False, "gapw_std", 1)


def example_cp2k_gw_gapw_hq_rks(cp2k_code):
    _example_cp2k_gw(cp2k_code, False, "gapw_hq", 0)


def example_cp2k_gw_gapw_hq_uks(cp2k_code):
    _example_cp2k_gw(cp2k_code, False, "gapw_hq", 1)


def example_cp2k_ic_gpw_std_rks(cp2k_code):
    _example_cp2k_gw(cp2k_code, True, "gpw_std", 0)


def example_cp2k_ic_gpw_std_uks(cp2k_code):
    _example_cp2k_gw(cp2k_code, True, "gpw_std", 1)


def example_cp2k_ic_gapw_std_rks(cp2k_code):
    _example_cp2k_gw(cp2k_code, True, "gapw_std", 0)


def example_cp2k_ic_gapw_std_uks(cp2k_code):
    _example_cp2k_gw(cp2k_code, True, "gapw_std", 1)


def example_cp2k_ic_gapw_hq_rks(cp2k_code):
    _example_cp2k_gw(cp2k_code, True, "gapw_hq", 0)


def example_cp2k_ic_gapw_hq_uks(cp2k_code):
    _example_cp2k_gw(cp2k_code, True, "gapw_hq", 1)


if __name__ == "__main__":
    for ic in [False, True]:
        for pc in ["gpw_std", "gapw_std", "gapw_hq"]:
            for mult in [0, 1]:
                print()
                print("####################################")
                print(f"#### ic={ic}; {pc}; mult={mult}")
                print("####################################")
                _example_cp2k_gw(load_code("cp2k@localhost"), ic, pc, mult)
