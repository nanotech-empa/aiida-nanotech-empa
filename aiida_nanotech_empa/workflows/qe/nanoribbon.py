import numpy as np
from aiida import engine, orm, plugins

from ...utils import common_utils

PwCalculation = plugins.CalculationFactory("quantumespresso.pw")
PpCalculation = plugins.CalculationFactory("quantumespresso.pp")
ProjwfcCalculation = plugins.CalculationFactory("quantumespresso.projwfc")


class NanoribbonWorkChain(engine.WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input(
            "optimize_cell",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(True),
            required=False,
        )
        spec.input(
            "max_kpoints",
            valid_type=orm.Int,
            default=lambda: orm.Int(120),
            required=False,
        )
        spec.input(
            "max_nodes", valid_type=orm.Int, default=lambda: orm.Int(24), required=False
        )
        spec.input(
            "mem_node", valid_type=orm.Int, default=lambda: orm.Int(64), required=False
        )
        spec.input("pw_code", valid_type=orm.Code)
        spec.input("pp_code", valid_type=orm.Code)
        spec.input("projwfc_code", valid_type=orm.Code)
        spec.input("structure", valid_type=orm.StructureData)
        spec.input(
            "tot_charge",
            valid_type=orm.Float,
            default=lambda: orm.Float(0.0),
            required=False,
        )
        spec.input(
            "precision",
            valid_type=orm.Float,
            default=lambda: orm.Float(1.0),
            required=False,
        )
        spec.input(
            "num_export_bands",
            valid_type=orm.Int,
            default=lambda: orm.Int(8),
            required=False,
        )
        spec.input(
            "pseudo_family",
            valid_type=orm.Str,
            required=True,
            help="An alternative to specifying the pseudo potentials manually in `pseudos`: one can specify the name "
            "of an existing pseudo potential family and the work chain will generate the pseudos automatically "
            "based on the input structure.",
        )
        # TODO: check why it does not work
        # spec.inputs("metadata.label", valid_type=six.string_types,
        #            default="NanoribbonWorkChain", non_db=True, help="Label of the work chain.")
        spec.outline(
            cls.run_cell_opt1,
            cls.run_cell_opt2,
            cls.run_scf,
            cls.run_export_hartree,
            cls.run_bands,
            cls.run_export_pdos,
            cls.run_bands_lowres,
            cls.prepare_export_orbitals,
            engine.while_(cls.should_run_export_orbitals)(
                cls.run_export_orbitals,
            ),
            cls.run_export_spinden,
            cls.run_closing,
        )
        spec.outputs.dynamic = True

        spec.exit_code(300, "CALC_FAILED", message="The calculation failed.")

    def run_cell_opt1(self):
        if self.inputs.optimize_cell.value:
            structure = self.inputs.structure
            return self._submit_pw_calc(
                structure,
                tot_charge=self.inputs.tot_charge.value,
                label="cell_opt1",
                runtype="vc-relax",
                precision=0.5,
                min_kpoints=int(1),
                max_nodes=self.inputs.max_nodes.value,
                mem_node=self.inputs.mem_node.value,
            )
        self.report("Skipping: cell_opt = False")
        return

    def run_cell_opt2(self):
        if self.inputs.optimize_cell.value:
            prev_calc = self.ctx.cell_opt1

            if not common_utils.check_if_calc_ok(self, prev_calc):
                return self.exit_codes.CALC_FAILED

            structure = prev_calc.outputs.output_structure
            return self._submit_pw_calc(
                structure,
                tot_charge=self.inputs.tot_charge.value,
                label="cell_opt2",
                runtype="vc-relax",
                precision=1.0,
                min_kpoints=int(1),
                max_nodes=self.inputs.max_nodes.value,
                mem_node=self.inputs.mem_node.value,
            )
        self.report("Skipping: cell_opt = False")
        return

    def run_scf(self):
        if self.inputs.optimize_cell.value:
            prev_calc = self.ctx.cell_opt2

            if not common_utils.check_if_calc_ok(self, prev_calc):
                return self.exit_codes.CALC_FAILED

            structure = prev_calc.outputs.output_structure
        else:
            structure = self.inputs.structure
        min_kpoints = min(int(10), self.inputs.max_kpoints.value)
        return self._submit_pw_calc(
            structure,
            tot_charge=self.inputs.tot_charge.value,
            label="scf",
            runtype="scf",
            precision=3.0,
            min_kpoints=min_kpoints,
            max_nodes=self.inputs.max_nodes.value,
            mem_node=self.inputs.mem_node.value,
            wallhours=4,
        )

    def run_export_hartree(self):
        self.report("Running pp.x to export hartree potential")
        label = "export_hartree"

        builder = PpCalculation.get_builder()
        builder.code = self.inputs.pp_code

        prev_calc = self.ctx.scf

        if not common_utils.check_if_calc_ok(self, prev_calc):
            return self.exit_codes.CALC_FAILED

        builder.parent_folder = prev_calc.outputs.remote_folder

        structure = prev_calc.inputs.structure
        cell_a = structure.cell[0][0]
        cell_b = structure.cell[1][1]
        cell_c = structure.cell[2][2]

        builder.parameters = orm.Dict(
            {
                "INPUTPP": {
                    "plot_num": 11,  # the V_bare + V_H potential
                },
                "PLOT": {
                    "iflag": 2,
                    "x0(1)": 0.0,
                    "x0(2)": 0.0,
                    "x0(3)": cell_c / cell_a,
                    # 3D vectors which determine the plotting plane
                    # in alat units)
                    "e1(1)": cell_a / cell_a,
                    "e1(2)": 0.0,
                    "e1(3)": 0.0,
                    "e2(1)": 0.0,
                    "e2(2)": cell_b / cell_a,
                    "e2(3)": 0.0,
                    "nx": 10,  # Number of points in the plane
                    "ny": 10,
                },
            }
        )

        natoms = len(prev_calc.inputs.structure.base.attributes.all["sites"])
        nnodes = min(
            self.inputs.max_nodes.value, (1 + int(natoms / self.inputs.mem_node.value))
        )
        # Reconsider the following lines, when https://gitlab.com/QEF/q-e/-/issues/221 is fixed.
        npools = 1
        # nnodes = int(prev_calc.attributes['resources']['num_machines'])
        # npools = int(prev_calc.inputs.settings.get_dict()['cmdline'][1])
        nproc_mach = builder.code.computer.get_default_mpiprocs_per_machine()

        builder.metadata.label = label

        builder.metadata.options = {
            "resources": {
                "num_machines": int(nnodes),
                "num_mpiprocs_per_machine": nproc_mach,
            },
            "max_wallclock_seconds": 1200,  # 30 minutes
            "withmpi": True,
        }
        builder.settings = orm.Dict({"cmdline": ["-npools", str(npools)]})
        return engine.ToContext(**{label: self.submit(builder)})

    def run_bands(self):
        if not common_utils.check_if_calc_ok(self, self.ctx.export_hartree):
            return self.exit_codes.CALC_FAILED

        prev_calc = self.ctx.scf
        structure = prev_calc.inputs.structure
        parent_folder = prev_calc.outputs.remote_folder
        min_kpoints = min(int(20), self.inputs.max_kpoints.value)
        return self._submit_pw_calc(
            structure,
            tot_charge=self.inputs.tot_charge.value,
            label="bands",
            parent_folder=parent_folder,
            runtype="bands",
            precision=4.0,
            min_kpoints=min_kpoints,
            max_nodes=self.inputs.max_nodes.value,
            mem_node=self.inputs.mem_node.value,
            wallhours=6,
        )

    def run_export_pdos(self):
        prev_calc = self.ctx.bands
        if not common_utils.check_if_calc_ok(self, prev_calc):
            return self.exit_codes.CALC_FAILED

        self.report("Running projwfc.x to export PDOS")
        label = "export_pdos"

        builder = ProjwfcCalculation.get_builder()
        builder.code = self.inputs.projwfc_code

        natoms = len(prev_calc.inputs.structure.base.attributes.all["sites"])
        nproc_mach = min(4, builder.code.computer.get_default_mpiprocs_per_machine())

        previous_nodes = int(prev_calc.base.attributes.all["resources"]["num_machines"])
        previous_pools = int(prev_calc.inputs.parallelization.get_dict()["npool"])
        if natoms < 60:
            nnodes = min(int(2), previous_nodes)
            npools = min(int(2), previous_pools)
        elif natoms < int(120):
            nnodes = min(int(4), previous_nodes)
            npools = min(int(4), previous_pools)
        else:
            nnodes = previous_nodes
            npools = previous_pools
            nproc_mach = builder.code.computer.get_default_mpiprocs_per_machine()

        nhours = 24
        builder.parent_folder = prev_calc.outputs.remote_folder

        # use the same number of pools as in bands calculation
        builder.parameters = orm.Dict(
            {
                "projwfc": {
                    "ngauss": 1,
                    "degauss": 0.007,
                    "DeltaE": 0.01,
                    "filproj": "projection.out",
                },
            }
        )

        builder.metadata.label = label
        builder.metadata.options = {
            "resources": {
                "num_machines": int(nnodes),
                "num_mpiprocs_per_machine": nproc_mach,
            },
            "max_wallclock_seconds": nhours * 60 * 60,  # hours
            "withmpi": True,
        }

        builder.settings = orm.Dict(
            {
                "additional_retrieve_list": [
                    "./out/aiida.save/*.xml",
                    "*_up",
                    "*_down",
                    "*_tot",
                ],
                "cmdline": ["-npools", str(npools)],
            }
        )

        future = self.submit(builder)
        return engine.ToContext(**{label: future})

    def run_bands_lowres(self):
        if not common_utils.check_if_calc_ok(self, self.ctx.export_pdos):
            return self.exit_codes.CALC_FAILED

        self.report("Running bands with fewer kpt to export KS")

        prev_calc = self.ctx.scf
        structure = prev_calc.inputs.structure
        parent_folder = prev_calc.outputs.remote_folder
        min_kpoints = min(int(12), self.inputs.max_kpoints.value)
        return self._submit_pw_calc(
            structure,
            tot_charge=self.inputs.tot_charge.value,
            label="bands_lowres",
            parent_folder=parent_folder,
            runtype="bands",
            precision=0.0,
            min_kpoints=min_kpoints,
            max_nodes=self.inputs.max_nodes.value,
            mem_node=self.inputs.mem_node.value,
            wallhours=2,
        )

    def prepare_export_orbitals(self):
        prev_calc = self.ctx.bands_lowres
        if not common_utils.check_if_calc_ok(self, prev_calc):
            return self.exit_codes.CALC_FAILED

        self.report("Getting ready to export KS orbitals.")

        self.ctx.export_orbitals_parameters = {
            "INPUTPP": {
                # contribution of a selected wavefunction
                # to charge density
                "plot_num": 7,
                "kpoint(1)": int(1),
                "kpoint(2)": int(
                    prev_calc.res.number_of_k_points
                    * prev_calc.res.number_of_spin_components
                ),
                "kband(1)": 0,
                "kband(2)": 0,
            },
            "PLOT": {
                "iflag": 3,  # 3D plot
            },
        }

        nhours = int(2 + min(22, 2 * int(prev_calc.res.volume / 1500)))
        # Reconsider the following lines, when https://gitlab.com/QEF/q-e/-/issues/221 is fixed.
        natoms = len(prev_calc.inputs.structure.base.attributes.all["sites"])
        nnodes = min(
            self.inputs.max_nodes.value, (1 + int(natoms / self.inputs.mem_node.value))
        )
        npools = 1
        self.ctx.export_orbitals_options = {
            "resources": {
                "num_machines": int(nnodes),
                "num_mpiprocs_per_machine": self.inputs.pp_code.computer.get_default_mpiprocs_per_machine(),
            },
            "max_wallclock_seconds": nhours * 60 * 60,
            # Add the post-processing python scripts.
            "withmpi": True,
            "parser_name": "nanotech_empa.pp",
        }
        self.ctx.export_orbitals_settings = orm.Dict(
            {"cmdline": ["-npools", str(npools)]}
        )

        kband1 = max(
            int(prev_calc.res.number_of_electrons / 2)
            - int(self.inputs.num_export_bands.value / 2)
            + 1,
            1,
        )
        self.ctx.first_band = kband1
        self.ctx.export_orbitals_band_number = kband1

    def should_run_export_orbitals(self):
        prev_calc = self.ctx.bands_lowres
        kband2 = min(
            int(prev_calc.res.number_of_electrons / 2)
            + int(self.inputs.num_export_bands.value / 2),
            int(prev_calc.res.number_of_bands),
        )
        return self.ctx.export_orbitals_band_number <= kband2

    def run_export_orbitals(self):
        # Check if the previous calculation has finished successfully.
        if self.ctx.export_orbitals_band_number == self.ctx.first_band:
            to_check = "bands_lowres"
        else:
            to_check = f"export_orbitals_{self.ctx.export_orbitals_band_number-1}"
        if not common_utils.check_if_calc_ok(self, getattr(self.ctx, to_check)):
            return self.exit_codes.CALC_FAILED

        self.report("Running pp.x to export KS orbitals")

        builder = PpCalculation.get_builder()
        builder.code = self.inputs.pp_code
        prev_calc = self.ctx.bands_lowres
        builder.parent_folder = prev_calc.outputs.remote_folder
        builder.metadata.label = "export_orbitals"
        builder.metadata.options = self.ctx.export_orbitals_options
        builder.settings = self.ctx.export_orbitals_settings

        # Modifying the band number.
        self.ctx.export_orbitals_parameters["INPUTPP"][
            "kband(1)"
        ] = self.ctx.export_orbitals_band_number
        self.ctx.export_orbitals_parameters["INPUTPP"][
            "kband(2)"
        ] = self.ctx.export_orbitals_band_number
        builder.parameters = orm.Dict(self.ctx.export_orbitals_parameters)

        # Running the calculation.
        running = self.submit(builder)
        label = f"export_orbitals_{self.ctx.export_orbitals_band_number}"
        self.ctx.export_orbitals_band_number += 1
        return engine.ToContext(**{label: running})

    def run_export_spinden(self):
        self.report("Running pp.x to compute spinden")
        label = "export_spinden"
        last_ks = f"export_orbitals_{self.ctx.export_orbitals_band_number - 1}"

        if not common_utils.check_if_calc_ok(self, getattr(self.ctx, last_ks)):
            return self.exit_codes.CALC_FAILED

        builder = PpCalculation.get_builder()
        builder.code = self.inputs.pp_code
        nproc_mach = builder.code.computer.get_default_mpiprocs_per_machine()
        prev_calc = self.ctx.scf
        builder.parent_folder = prev_calc.outputs.remote_folder

        nspin = prev_calc.res.number_of_spin_components
        natoms = len(prev_calc.inputs.structure.base.attributes.all["sites"])
        nnodes = min(
            self.inputs.max_nodes.value, (1 + int(natoms / self.inputs.mem_node.value))
        )
        # Reconsider the following lines, when https://gitlab.com/QEF/q-e/-/issues/221 is fixed.
        npools = 1
        if nspin == 1:
            self.report("Skipping, got only one spin channel")
            return

        builder.parameters = orm.Dict(
            {
                "INPUTPP": {
                    "plot_num": 6,  # spin polarization (rho(up)-rho(down))
                },
                "PLOT": {
                    "iflag": 3,  # 3D plot
                },
            }
        )

        builder.metadata.label = label
        builder.metadata.options = {
            "resources": {
                "num_machines": int(nnodes),
                "num_mpiprocs_per_machine": nproc_mach,
            },
            "max_wallclock_seconds": 30 * 60,  # 30 minutes
            "withmpi": True,
            "parser_name": "nanotech_empa.pp",
        }

        builder.settings = orm.Dict({"cmdline": ["-npools", str(npools)]})

        future = self.submit(builder)
        return engine.ToContext(**{label: future})

    def run_closing(self):
        self.report("Running final check")
        # Getting and checking the previous calculation.
        nspin = self.ctx.scf.res.number_of_spin_components

        # Check if previous calc was okay.
        if nspin > 1:
            prev_calc = self.ctx.export_spinden
            if not common_utils.check_if_calc_ok(self, prev_calc):
                return self.exit_codes.CALC_FAILED

            self.out(
                "spin_density_arraydata", self.ctx.export_spinden.outputs.output_data
            )

        self.report("END of workchain")
        return

    def _submit_pw_calc(
        self,
        structure,
        tot_charge,
        label,
        runtype,
        precision,
        min_kpoints,
        max_nodes,
        mem_node,
        wallhours=24,
        parent_folder=None,
    ):
        self.report("Running pw.x for " + label)
        builder = PwCalculation.get_builder()

        builder.code = self.inputs.pw_code
        builder.structure = structure
        builder.parameters = self._get_parameters(structure, tot_charge, runtype, label)

        # Loading family from input.
        family_pseudo = orm.load_group(self.inputs.pseudo_family.value)
        builder.pseudos = family_pseudo.get_pseudos(structure=structure)

        if parent_folder:
            builder.parent_folder = parent_folder

        # kpoints
        cell_a = builder.structure.cell[0][0]
        precision *= self.inputs.precision.value
        nkpoints = max(min_kpoints, int(30 * 2.5 / cell_a * precision))

        if self.inputs.max_kpoints.value < nkpoints:
            self.report(
                "max kpoints exceeded, instead of  "
                + str(nkpoints)
                + " k-points, using "
                + str(min_kpoints)
            )
            nkpoints = self.inputs.max_kpoints.value  # for test runs minimal memory

        use_symmetry = runtype != "bands"
        kpoints = self._get_kpoints(nkpoints, use_symmetry=use_symmetry)
        builder.kpoints = kpoints

        # parallelization settings
        # Temporary double pools in case of spin
        spinpools = int(1)
        start_mag = self._get_magnetization(structure)
        if any(m != 0 for m in start_mag.values()):
            spinpools = int(2)

        natoms = len(structure.sites)
        max_npools = spinpools * min(1 + int(nkpoints / 4), int(6))
        nnodes_base = min(max_nodes, (1 + int(natoms / mem_node)))

        guess_nnodes = max_npools * nnodes_base
        if guess_nnodes <= max_nodes:
            nnodes = guess_nnodes
            npools = max_npools
        else:
            nnodes = max_nodes
            npools = 1
            cpus_per_node = builder.code.computer.get_default_mpiprocs_per_machine()
            for i in range(max_npools):
                if nnodes * cpus_per_node % (i + 1) == 0:
                    npools = i + 1
        if (
            nnodes == 1
            and builder.code.computer.get_default_mpiprocs_per_machine() == 1
        ):
            npools = 1
        builder.metadata.label = label
        # nnodes.store
        builder.metadata.options = {
            "resources": {"num_machines": int(nnodes)},
            "withmpi": True,
            "max_wallclock_seconds": wallhours * 60 * 60,
        }

        builder.parallelization = orm.Dict({"npool": int(npools)})

        future = self.submit(builder)
        return engine.ToContext(**{label: future})

    def _get_parameters(self, structure, tot_charge, runtype, label):
        params = {
            "CONTROL": {
                "calculation": runtype,
                "wf_collect": True,
                "forc_conv_thr": 0.0001,
                "nstep": 500,
            },
            "SYSTEM": {
                "ecutwfc": 50.0,
                "ecutrho": 400.0,
                "occupations": "smearing",
                "degauss": 0.001,
                "tot_charge": tot_charge,
            },
            "ELECTRONS": {
                "conv_thr": 1.0e-8,
                "mixing_beta": 0.25,
                "electron_maxstep": 50,
                "scf_must_converge": False,
            },
        }

        if label == "cell_opt1":
            params["CONTROL"]["forc_conv_thr"] = 0.0005
        if runtype == "vc-relax":
            # In y and z direction there is only vacuum.
            params["CELL"] = {"cell_dofree": "x"}

        start_mag = self._get_magnetization(structure)
        if any(m != 0 for m in start_mag.values()):
            params["SYSTEM"]["nspin"] = 2
            params["SYSTEM"]["starting_magnetization"] = start_mag

        return orm.Dict(params)

    def _get_kpoints(self, nx, use_symmetry=True):
        nx = max(1, nx)

        kpoints = orm.KpointsData()
        if use_symmetry:
            kpoints.set_kpoints_mesh([nx, 1, 1], offset=[0.0, 0.0, 0.0])
        else:
            # List kpoints explicitly.
            points = [[r, 0.0, 0.0] for r in np.linspace(0, 0.5, nx)]
            kpoints.set_kpoints(points)
        return kpoints

    def _get_magnetization(self, structure):
        start_mag = {}
        for i in structure.kinds:
            if i.name.endswith("1"):
                start_mag[i.name] = 1.0
            elif i.name.endswith("2"):
                start_mag[i.name] = -1.0
            else:
                start_mag[i.name] = 0.0
        return start_mag
