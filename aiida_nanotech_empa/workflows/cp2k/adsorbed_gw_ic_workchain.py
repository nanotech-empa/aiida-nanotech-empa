from aiida_nanotech_empa.utils import common_utils

import numpy as np

from aiida.engine import WorkChain, ExitCode, calcfunction, ToContext, if_
from aiida.orm import Int, Str, Bool, Code, List, Float, Dict
from aiida.orm import StructureData

from aiida.plugins import WorkflowFactory

Cp2kMoleculeGwWorkChain = WorkflowFactory('nanotech_empa.cp2k.molecule_gw')
Cp2kMoleculeOptWorkChain = WorkflowFactory('nanotech_empa.cp2k.molecule_opt')


def _extract_image_plane(ase_geom):
    au_layer_dz = 1.0  # ang
    au_atoms = ase_geom[np.array(ase_geom.get_chemical_symbols()) == 'Au']
    au_top_layer = au_atoms[
        np.abs(np.max(au_atoms.positions[:, 2]) -
               au_atoms.positions[:, 2]) < au_layer_dz]
    gold_surf_z = np.mean(au_top_layer.positions[:, 2])

    imag_plane_z = gold_surf_z + 1.42  # Kharche et al

    return imag_plane_z


@calcfunction
def extract_image_plane(adsorbed_structure):
    ase_geom = adsorbed_structure.get_ase()
    imag_plane_z = _extract_image_plane(ase_geom)
    return Float(imag_plane_z)


@calcfunction
def extract_molecule(adsorbed_structure):
    ase_geom = adsorbed_structure.get_ase()
    imag_plane_z = _extract_image_plane(ase_geom)
    molecule_atoms = ase_geom[ase_geom.positions[:, 2] > imag_plane_z]
    return StructureData(ase=molecule_atoms)


@calcfunction
def extract_molecule_mags_per_site(adsorbed_structure, mag_per_site):
    ase_geom = adsorbed_structure.get_ase()
    imag_plane_z = _extract_image_plane(ase_geom)

    mps = []
    if list(mag_per_site):
        mps = [
            m for at, m in zip(ase_geom, list(mag_per_site))
            if at.position[2] > imag_plane_z
        ]
    return List(list=mps)


@calcfunction
def calc_gw_ic_parameters(gw_params, ic_params):

    gw_evals = gw_params['gw_eval']
    ic_deltas = ic_params['ic_delta']
    homo_inds = gw_params['homo']

    homo_ens = []
    lumo_ens = []
    gw_ic_levels = []

    for i_spin in range(len(gw_evals)):
        h_i = homo_inds[i_spin]
        gw_ic = [
            gw + ic for gw, ic in zip(gw_evals[i_spin], ic_deltas[i_spin])
        ]

        # sort occupied and unoccupied energies separately, as the IC can change the order
        gw_ic[0:h_i + 1] = sorted(gw_ic[0:h_i + 1])
        gw_ic[h_i + 1:] = sorted(gw_ic[h_i + 1:])

        gw_ic_levels.append(gw_ic)

        if h_i != -1:
            homo_ens.append(gw_ic[h_i])
            lumo_ens.append(gw_ic[h_i + 1])

    gw_ic_params = {
        'gw_ic_levels': gw_ic_levels,
        'gw_ic_gap': min(lumo_ens) - max(homo_ens),
        'occ': gw_params['occ'],
        'homo': gw_params['homo'],
        'mo': gw_params['mo'],
        'gw_levels': gw_params['gw_eval'],
        'ic_deltas': ic_params['ic_delta'],
        'scf_levels': gw_params['g0w0_e_scf'],
    }

    return Dict(dict=gw_ic_params)


class Cp2kAdsorbedGwIcWorkChain(WorkChain):
    """
    WorkChain to run GW and IC for an adosrbed system

    Currently only Au(111) substrate is supported (and assumed)
    """
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("code", valid_type=Code)

        spec.input("ads_struct",
                   valid_type=StructureData,
                   help="Adsorbed molecule on a metal slab.")
        spec.input("protocol",
                   valid_type=Str,
                   default=lambda: Str('gapw_std'),
                   required=False,
                   help="Protocol supported by the GW workchain.")
        spec.input("multiplicity",
                   valid_type=Int,
                   default=lambda: Int(0),
                   required=False)
        spec.input("magnetization_per_site",
                   valid_type=List,
                   default=lambda: List(list=[]),
                   required=False)
        spec.input("walltime_seconds",
                   valid_type=Int,
                   default=lambda: Int(600),
                   required=False)
        spec.input("max_nodes",
                   valid_type=Int,
                   default=lambda: Int(2056),
                   required=False)
        spec.input("debug",
                   valid_type=Bool,
                   default=lambda: Bool(False),
                   required=False,
                   help="Run with fast parameters for debugging.")

        spec.input("geometry_mode",
                   valid_type=Str,
                   default=lambda: Str("ads_geo"),
                   required=False,
                   help="Possibilities: ads_geo, gas_opt")
        spec.input(
            "geometry_opt_mult",
            valid_type=Int,
            default=lambda: Int(0),
            required=False,
            help="Multiplicity in case of gas optimization is selected.")

        spec.outline(cls.setup,
                     if_(cls.gas_opt_selected)(cls.gas_opt, cls.check_gas_opt),
                     cls.ic, cls.gw, cls.finalize)
        spec.outputs.dynamic = True

        spec.exit_code(
            390,
            "ERROR_TERMINATION",
            message="One or more steps of the work chain failed.",
        )

    def setup(self):
        self.report("Inspecting input and setting up things.")

        n_atoms = len(self.inputs.ads_struct.get_ase())
        n_mags = len(list(self.inputs.magnetization_per_site))
        if n_mags not in (0, n_atoms):
            self.report(
                "If set, magnetization_per_site needs a value for every atom.")
            return self.exit_codes.ERROR_TERMINATION  # pylint: disable=no-member

        self.ctx.image_plane_z = extract_image_plane(self.inputs.ads_struct)
        self.ctx.mol_struct = extract_molecule(self.inputs.ads_struct)
        self.ctx.mol_mag_per_site = extract_molecule_mags_per_site(
            self.inputs.ads_struct, self.inputs.magnetization_per_site)

        return ExitCode(0)

    def gas_opt_selected(self):
        return self.inputs.geometry_mode.value == "gas_opt"

    def gas_opt(self):
        builder = Cp2kMoleculeOptWorkChain.get_builder()
        builder.code = self.inputs.code
        builder.structure = self.ctx.mol_struct
        builder.multiplicity = self.inputs.geometry_opt_mult
        builder.magnetization_per_site = self.ctx.mol_mag_per_site
        builder.vdw = Bool(True)
        builder.walltime_seconds = self.inputs.walltime_seconds
        builder.debug = self.inputs.debug

        builder.metadata.description = "gas_opt"
        submitted_node = self.submit(builder)
        return ToContext(gas_opt=submitted_node)

    def check_gas_opt(self):
        if not common_utils.check_if_calc_ok(self, self.ctx.gas_opt):
            return self.exit_codes.ERROR_TERMINATION  # pylint: disable=no-member
        # set the optimized geometrical center to the adsorbed conf geometrical center
        ads_mol_ase = self.ctx.mol_struct.get_ase()
        ads_mol_geo_center = np.mean(ads_mol_ase.positions, axis=0)
        gas_opt_ase = self.ctx.gas_opt.outputs.output_structure.get_ase()
        gas_opt_geo_center = np.mean(gas_opt_ase.positions, axis=0)
        gas_opt_ase.positions += ads_mol_geo_center - gas_opt_geo_center
        self.ctx.mol_struct = StructureData(ase=gas_opt_ase)
        return ExitCode(0)

    def ic(self):
        self.report("Submitting IC.")

        # Run IC first, because it has a higher chance of failure

        builder = Cp2kMoleculeGwWorkChain.get_builder()
        builder.code = self.inputs.code
        builder.protocol = self.inputs.protocol
        builder.structure = self.ctx.mol_struct
        builder.magnetization_per_site = self.ctx.mol_mag_per_site
        builder.multiplicity = self.inputs.multiplicity

        builder.debug = self.inputs.debug
        builder.walltime_seconds = self.inputs.walltime_seconds
        builder.max_nodes = self.inputs.max_nodes

        builder.image_charge = Bool(True)
        builder.z_ic_plane = self.ctx.image_plane_z

        builder.options = Dict(
            dict={
                'resources': {
                    'num_mpiprocs_per_machine': 1,
                    'num_cores_per_mpiproc': 6,
                }
            })

        builder.metadata.description = "ic"
        submitted_node = self.submit(builder)
        return ToContext(ic=submitted_node)

    def gw(self):

        if not common_utils.check_if_calc_ok(self, self.ctx.ic):
            return self.exit_codes.ERROR_TERMINATION  # pylint: disable=no-member

        self.report("Submitting GW.")

        builder = Cp2kMoleculeGwWorkChain.get_builder()
        builder.code = self.inputs.code
        builder.protocol = self.inputs.protocol
        builder.structure = self.ctx.mol_struct
        builder.magnetization_per_site = self.ctx.mol_mag_per_site
        builder.multiplicity = self.inputs.multiplicity

        builder.debug = self.inputs.debug
        builder.walltime_seconds = self.inputs.walltime_seconds
        builder.max_nodes = self.inputs.max_nodes

        builder.options = Dict(
            dict={
                'resources': {
                    'num_mpiprocs_per_machine': 1,
                    'num_cores_per_mpiproc': 6,
                }
            })

        builder.metadata.description = "gw"
        submitted_node = self.submit(builder)
        return ToContext(gw=submitted_node)

    def finalize(self):
        self.report("Finalizing...")

        if not common_utils.check_if_calc_ok(self, self.ctx.gw):
            return self.exit_codes.ERROR_TERMINATION  # pylint: disable=no-member

        gw_out_params = self.ctx.gw.outputs.gw_output_parameters
        ic_out_params = self.ctx.ic.outputs.gw_output_parameters

        self.out('gw_output_parameters', gw_out_params)

        self.out('ic_output_parameters', ic_out_params)

        self.out('gw_ic_parameters',
                 calc_gw_ic_parameters(gw_out_params, ic_out_params))

        return ExitCode(0)
