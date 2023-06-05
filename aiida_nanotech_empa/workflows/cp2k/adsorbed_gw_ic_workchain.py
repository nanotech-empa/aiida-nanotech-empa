import numpy as np
from aiida import engine, orm

from .geo_opt_workchain import Cp2kGeoOptWorkChain
from .molecule_gw_workchain import Cp2kMoleculeGwWorkChain

IC_PLANE_HEIGHTS = {
    "Au(111)": 1.42,  # Kharche J. Phys. Chem. Lett. 7, 1526–1533 (2016).
}


def geometrical_analysis(ase_geo, substr_elem):
    """Simple geometry analysis that returns in the case of
    1) an isolated molecule -> geometry, None
    2) adsorbed system -> molecular geometry, top substr. layer z
    """
    chem_symbols_arr = np.array(ase_geo.get_chemical_symbols())
    s_atoms = ase_geo[chem_symbols_arr == substr_elem]
    non_s_atoms = ase_geo[chem_symbols_arr != substr_elem]
    if len(s_atoms) == 0:
        return ase_geo, None
    layer_dz = 1.0  # ang
    substr_top_layer = s_atoms[
        np.abs(np.max(s_atoms.positions[:, 2]) - s_atoms.positions[:, 2]) < layer_dz
    ]
    surf_z = np.mean(substr_top_layer.positions[:, 2])

    mol_atoms = non_s_atoms[non_s_atoms.positions[:, 2] > surf_z]

    return mol_atoms, surf_z


@engine.calcfunction
def analyze_structure(structure, substrate, mag_per_site, ads_h=None):
    ase_geo = structure.get_ase()
    substr_elem = substrate.value.split("(")[0]

    mol_atoms, surf_z = geometrical_analysis(ase_geo, substr_elem)

    if surf_z is None:
        if ads_h is None:
            return engine.ExitCode(
                300, "Ads. height not specified for isolated molecule."
            )
        # Adsorption height is defined from the geometrical center of the molecule
        surf_z = np.mean(mol_atoms.positions[:, 2]) - ads_h.value

    else:
        # If you manually specify adsorption height, it will override the
        # height extracted from the geometry
        if ads_h is not None:
            surf_z = np.mean(mol_atoms.positions[:, 2]) - ads_h.value

    imag_plane_z = surf_z + IC_PLANE_HEIGHTS[substrate.value]

    mps = []
    if list(mag_per_site):
        mol_at_tuples = [
            (e, *np.round(p, 2))
            for e, p in zip(mol_atoms.get_chemical_symbols(), mol_atoms.positions)
        ]
        mps = [
            m
            for at, m in zip(ase_geo, list(mag_per_site))
            if (at.symbol, *np.round(at.position, 2)) in mol_at_tuples
        ]

    return {
        "mol_struct": orm.StructureData(ase=mol_atoms),
        "image_plane_z": orm.Float(imag_plane_z),
        "mol_mag_per_site": orm.List(mps),
    }


@engine.calcfunction
def calc_gw_ic_parameters(gw_params, ic_params):
    gw_evals = gw_params["gw_eval"]
    ic_deltas = ic_params["ic_delta"]
    homo_inds = gw_params["homo"]

    homo_ens = []
    lumo_ens = []
    gw_ic_levels = []

    for i_spin in range(len(gw_evals)):
        h_i = homo_inds[i_spin]
        gw_ic = [gw + ic for gw, ic in zip(gw_evals[i_spin], ic_deltas[i_spin])]

        # sort occupied and unoccupied energies separately, as the IC can change the order
        gw_ic[0 : h_i + 1] = sorted(gw_ic[0 : h_i + 1])
        gw_ic[h_i + 1 :] = sorted(gw_ic[h_i + 1 :])

        gw_ic_levels.append(gw_ic)

        if h_i != -1:
            homo_ens.append(gw_ic[h_i])
            lumo_ens.append(gw_ic[h_i + 1])

    gw_ic_params = {
        "gw_ic_levels": gw_ic_levels,
        "gw_ic_gap": min(lumo_ens) - max(homo_ens),
        "occ": gw_params["occ"],
        "homo": gw_params["homo"],
        "mo": gw_params["mo"],
        "gw_levels": gw_params["gw_eval"],
        "ic_deltas": ic_params["ic_delta"],
        "scf_levels": gw_params["g0w0_e_scf"],
    }

    return orm.Dict(gw_ic_params)


class Cp2kAdsorbedGwIcWorkChain(engine.WorkChain):
    """
    WorkChain to run GW and IC for an adsorbed system

    Two different ways to run:
    1) geometry of a molecule adsorbed on a substrate
    2) isolated molecule & adsorption height
    """

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("code", valid_type=orm.Code)

        spec.input(
            "structure",
            valid_type=orm.StructureData,
            help="A molecule on a substrate or an isolated molecule.",
        )
        spec.input(
            "ads_height",
            valid_type=orm.Float,
            required=False,
            help=(
                "Adsoprtion height from the molecular geometrical center."
                "Required if an isolated molecule is specified."
            ),
        )
        spec.input(
            "substrate",
            valid_type=orm.Str,
            default=lambda: orm.Str("Au(111)"),
            help="Substrate type, determines the image charge plane.",
        )
        spec.input(
            "protocol",
            valid_type=orm.Str,
            default=lambda: orm.Str("gpw_std"),
            required=False,
            help="Protocol supported by the Cp2kMoleculeGwWorkChain.",
        )
        spec.input(
            "multiplicity",
            valid_type=orm.Int,
            default=lambda: orm.Int(0),
            required=False,
        )
        spec.input(
            "magnetization_per_site",
            valid_type=orm.List,
            default=lambda: orm.List(list=[]),
            required=False,
        )
        spec.input_namespace(
            "options",
            valid_type=dict,
            non_db=True,
            required=False,
            help="Define options for the cacluations: walltime, memory, CPUs, etc.",
        )
        spec.input(
            "options.scf",
            valid_type=dict,
            non_db=True,
            required=False,
            help="Define options for the SCF cacluation: walltime, memory, CPUs, etc.",
        )
        spec.input(
            "options.gw",
            valid_type=dict,
            non_db=True,
            required=False,
            help="Define options for the GW cacluation: walltime, memory, CPUs, etc.",
        )
        spec.input(
            "options.ic",
            valid_type=dict,
            non_db=True,
            required=False,
            help="Define options for the GW cacluation: walltime, memory, CPUs, etc.",
        )
        spec.input(
            "geometry_mode",
            valid_type=orm.Str,
            default=lambda: orm.Str("ads_geo"),
            required=False,
            help="Possibilities: ads_geo, gas_opt",
        )

        spec.outline(
            cls.setup,
            engine.if_(cls.gas_opt_selected)(cls.gas_opt, cls.check_gas_opt),
            cls.ic,
            cls.gw,
            cls.finalize,
        )
        spec.outputs.dynamic = True

        spec.exit_code(
            380,
            "ERROR_SUBSTR_NOT_SUPPORTED",
            message="Specified substrate is not supported.",
        )
        spec.exit_code(
            381,
            "ERROR_STRUCTURE_ANALYSIS",
            message="Structure analysis failed.",
        )
        spec.exit_code(
            390,
            "ERROR_TERMINATION",
            message="One or more steps of the work chain failed.",
        )

    def setup(self):
        self.report("Inspecting input and setting up things.")

        n_atoms = len(self.inputs.structure.get_ase())
        n_mags = len(list(self.inputs.magnetization_per_site))
        if n_mags not in (0, n_atoms):
            self.report("If set, magnetization_per_site needs a value for every atom.")
            return self.exit_codes.ERROR_TERMINATION

        if self.inputs.substrate.value not in IC_PLANE_HEIGHTS:
            return self.exit_codes.ERROR_SUBSTR_NOT_SUPPORTED

        an_out = analyze_structure(
            self.inputs.structure,
            self.inputs.substrate,
            self.inputs.magnetization_per_site,
            None if "ads_height" not in self.inputs else self.inputs.ads_height,
        )

        if "mol_struct" not in an_out:
            self.report("Structure analyis failed.")
            return self.exit_codes.ERROR_STRUCTURE_ANALYSIS

        self.ctx.mol_struct = an_out["mol_struct"]
        self.ctx.image_plane_z = an_out["image_plane_z"]
        self.ctx.mol_mag_per_site = an_out["mol_mag_per_site"]

        return engine.ExitCode(0)

    def gas_opt_selected(self):
        return self.inputs.geometry_mode.value == "gas_opt"

    def gas_opt(self):
        builder = Cp2kGeoOptWorkChain.get_builder()
        builder.code = self.inputs.code
        builder.structure = self.ctx.mol_struct
        builder.multiplicity = self.inputs.multiplicity
        builder.magnetization_per_site = self.ctx.mol_mag_per_site
        builder.vdw = orm.Bool(True)
        builder.protocol = orm.Str("standard")
        builder.options = self.inputs.options.scf
        builder.metadata.description = "gas_opt"
        submitted_node = self.submit(builder)
        return engine.ToContext(gas_opt=submitted_node)

    def check_gas_opt(self):
        if not self.ctx.gas_opt.is_finished_ok:
            return self.exit_codes.ERROR_TERMINATION
        # set the optimized geometrical center to the adsorbed conf geometrical center
        ads_mol_ase = self.ctx.mol_struct.get_ase()
        ads_mol_geo_center = np.mean(ads_mol_ase.positions, axis=0)
        gas_opt_ase = self.ctx.gas_opt.outputs.output_structure.get_ase()
        gas_opt_geo_center = np.mean(gas_opt_ase.positions, axis=0)
        gas_opt_ase.positions += ads_mol_geo_center - gas_opt_geo_center
        self.ctx.mol_struct = orm.StructureData(ase=gas_opt_ase)
        return engine.ExitCode(0)

    def ic(self):
        self.report("Submitting IC.")

        # Run IC first, because it has a higher chance of failure

        builder = Cp2kMoleculeGwWorkChain.get_builder()
        builder.code = self.inputs.code
        builder.protocol = self.inputs.protocol
        builder.structure = self.ctx.mol_struct
        builder.magnetization_per_site = self.ctx.mol_mag_per_site
        builder.multiplicity = self.inputs.multiplicity
        builder.run_image_charge = orm.Bool(True)
        builder.z_ic_plane = self.ctx.image_plane_z
        builder.options.scf = self.inputs.options.scf
        builder.options.gw = self.inputs.options.ic
        builder.metadata.description = "ic"
        submitted_node = self.submit(builder)
        return engine.ToContext(ic=submitted_node)

    def gw(self):
        if not self.ctx.ic.is_finished_ok:
            return self.exit_codes.ERROR_TERMINATION

        self.report("Submitting GW.")

        builder = Cp2kMoleculeGwWorkChain.get_builder()
        builder.code = self.inputs.code
        builder.protocol = self.inputs.protocol
        builder.structure = self.ctx.mol_struct
        builder.magnetization_per_site = self.ctx.mol_mag_per_site
        builder.multiplicity = self.inputs.multiplicity
        builder.options.scf = self.inputs.options.scf
        builder.options.gw = self.inputs.options.gw
        builder.metadata.description = "gw"
        submitted_node = self.submit(builder)
        return engine.ToContext(gw=submitted_node)

    def finalize(self):
        self.report("Finalizing...")
        if not self.ctx.gw.is_finished_ok:
            return self.exit_codes.ERROR_TERMINATION

        gw_out_params = self.ctx.gw.outputs.gw_output_parameters
        ic_out_params = self.ctx.ic.outputs.gw_output_parameters

        self.out("gw_output_parameters", gw_out_params)

        self.out("ic_output_parameters", ic_out_params)

        self.out(
            "gw_ic_parameters", calc_gw_ic_parameters(gw_out_params, ic_out_params)
        )

        # Add the workchain pk to the input structure extras
        extras_label = "Cp2kAdsorbedGwIcWorkChain_pks"
        if extras_label not in self.inputs.structure.base.extras.all:
            extras_list = []
        else:
            extras_list = self.inputs.structure.base.extras.all[extras_label]
        extras_list.append(self.node.pk)
        self.inputs.structure.base.extras.set(extras_label, extras_list)

        return engine.ExitCode(0)
