import pathlib

import numpy as np
from aiida.engine import run_get_node
from aiida.orm import (
    Bool,
    Dict,
    Int,
    List,
    SinglefileData,
    Str,
    StructureData,
    load_code,
)
from aiida.plugins import CalculationFactory, WorkflowFactory
from ase import Atoms

from aiida_nanotech_empa.workflows.cp2k import cp2k_utils
from aiida_nanotech_empa.workflows.cp2k.molecule_opt_gw_workchain import (
    geo_opt_dft_params,
)

Cp2kCalculation = CalculationFactory("cp2k")
Cp2kMoleculeOptGwWorkChain = WorkflowFactory("nanotech_empa.cp2k.mol_opt_gw")


def _h2_structure():
    ase_geom = Atoms("HH", positions=[[0, 0, 0], [0.75, 0, 0]], cell=[4.0, 4.0, 4.0])
    return StructureData(ase=ase_geom)


def _geo_opt_cp2k_builder(cp2k_code):
    structure = _h2_structure()
    dft_params = geo_opt_dft_params(Int(0), List(list=[]), structure)
    input_dict = cp2k_utils.load_protocol("geo_opt_protocol.yml", "debug")

    input_dict["FORCE_EVAL"]["DFT"]["CHARGE"] = dft_params["charge"]

    structure_with_tags, kinds_dict = cp2k_utils.determine_kinds(structure)
    ase_atoms = structure_with_tags.get_ase()
    extra_cell = 5.0
    ase_atoms.cell = 2 * (np.ptp(ase_atoms.positions, axis=0)) + extra_cell
    ase_atoms.center()

    input_dict["FORCE_EVAL"]["SUBSYS"]["CELL"]["PERIODIC"] = "NONE"
    input_dict["FORCE_EVAL"]["DFT"]["POISSON"]["PERIODIC"] = "NONE"
    input_dict["FORCE_EVAL"]["DFT"]["POISSON"]["POISSON_SOLVER"] = "MT"
    input_dict["FORCE_EVAL"]["DFT"]["MGRID"]["CUTOFF"] = cp2k_utils.get_cutoff(
        structure=structure
    )
    cp2k_utils.dict_merge(
        input_dict, cp2k_utils.get_kinds_section(kinds_dict, protocol="gpw")
    )

    data_dir = (
        pathlib.Path(__file__).parents[2]
        / "aiida_nanotech_empa/workflows/cp2k/data"
    )

    builder = Cp2kCalculation.get_builder()
    builder.code = cp2k_code
    builder.structure = StructureData(ase=ase_atoms)
    builder.file = {
        "basis": SinglefileData(file=data_dir / "BASIS_MOLOPT"),
        "pseudo": SinglefileData(file=data_dir / "POTENTIAL"),
    }
    builder.parameters = Dict(input_dict)
    builder.metadata.options = {
        "max_wallclock_seconds": 600,
        "resources": {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
            "num_cores_per_mpiproc": 1,
        },
        "parser_name": "cp2k_advanced_parser",
    }
    builder.metadata.dry_run = True
    builder.metadata.store_provenance = False
    return builder


def _example_cp2k_mol_opt_gw(cp2k_code, multiplicity=0, mag_list=None):
    builder = Cp2kMoleculeOptGwWorkChain.get_builder()

    builder.metadata.description = "H2 gas"
    builder.code = cp2k_code

    builder.structure = _h2_structure()
    if mag_list is not None:
        builder.magnetization_per_site = List(mag_list)

    builder.protocol = Str("gpw_std")
    builder.multiplicity = Int(multiplicity)
    builder.debug = Bool(True)

    builder.geo_opt = Bool(False)

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
    _, calc_node = run_get_node(_geo_opt_cp2k_builder(cp2k_code))
    input_file = pathlib.Path(calc_node.dry_run_info["folder"]) / "aiida.inp"
    input_text = input_file.read_text()

    assert "RUN_TYPE GEO_OPT" in input_text
    assert "PERIODIC NONE" in input_text
    assert "POISSON_SOLVER MT" in input_text
    assert "UKS .FALSE." in input_text
    assert "MULTIPLICITY 0" in input_text


def example_cp2k_mol_opt_gw_no_geo_opt(cp2k_code):
    _example_cp2k_mol_opt_gw(cp2k_code, multiplicity=1, mag_list=[-1, 1])


if __name__ == "__main__":
    print("# Run geometry optimization and then run GW #")
    example_cp2k_mol_opt_gw_geo_opt(load_code("cp2k@localhost"))
    print("# Run GW only #")
    example_cp2k_mol_opt_gw_no_geo_opt(load_code("cp2k@localhost"))
