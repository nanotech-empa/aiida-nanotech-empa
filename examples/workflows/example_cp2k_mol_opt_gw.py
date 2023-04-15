from aiida.engine import run_get_node
from aiida.orm import Bool, Int, List, Str, StructureData, load_code
from aiida.plugins import WorkflowFactory
from ase import Atoms

Cp2kMoleculeOptGwWorkChain = WorkflowFactory("nanotech_empa.cp2k.mol_opt_gw")


def _example_cp2k_mol_opt_gw(cp2k_code, geo_opt):
    builder = Cp2kMoleculeOptGwWorkChain.get_builder()

    builder.metadata.description = "H2 gas"
    builder.code = cp2k_code

    ase_geom = Atoms("HH", positions=[[0, 0, 0], [0.75, 0, 0]], cell=[4.0, 4.0, 4.0])
    mag_list = [-1, 1]

    builder.structure = StructureData(ase=ase_geom)
    builder.magnetization_per_site = List(mag_list)

    builder.protocol = Str("gpw_std")
    builder.multiplicity = Int(1)

    builder.geo_opt = Bool(False)
    if geo_opt:
        builder.geo_opt = Bool(True)

    builder.options.geo_opt = {
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

    gw_res = dict(calc_node.outputs.gw_output_parameters)
    print()
    for k in gw_res:
        print(f"  {k}: {gw_res[k]}")
    print()


def example_cp2k_mol_opt_gw_geo_opt(cp2k_code):
    _example_cp2k_mol_opt_gw(cp2k_code, geo_opt=True)


def example_cp2k_mol_opt_gw_no_geo_opt(cp2k_code):
    _example_cp2k_mol_opt_gw(cp2k_code, geo_opt=False)


if __name__ == "__main__":
    print("# Run geometry optimization and then run GW #")
    example_cp2k_mol_opt_gw_geo_opt(load_code("cp2k@localhost"))
    print("# Run GW only #")
    example_cp2k_mol_opt_gw_no_geo_opt(load_code("cp2k@localhost"))
