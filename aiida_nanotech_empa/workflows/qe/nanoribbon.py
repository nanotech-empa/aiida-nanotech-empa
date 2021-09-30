# pylint: disable=too-many-locals,inconsistent-return-statements

import numpy as np

# AiiDA imports
from aiida.orm import Bool, Code, Dict, Float, Int, KpointsData, Str, StructureData
from aiida.engine import WorkChain, ToContext, while_
#from aiida.orm.nodes.data.upf import get_pseudos_dict, get_pseudos_from_structure

# aiida_quantumespresso imports
from aiida_quantumespresso.calculations.pw import PwCalculation
from aiida_quantumespresso.calculations.pp import PpCalculation
from aiida_quantumespresso.calculations.projwfc import ProjwfcCalculation
from aiida_quantumespresso.utils.pseudopotential import validate_and_prepare_pseudos_inputs


class NanoribbonWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super(NanoribbonWorkChain, cls).define(spec)
        spec.input("optimize_cell",
                   valid_type=Bool,
                   default=lambda: Bool(True),
                   required=False)
        spec.input("max_kpoints",
                   valid_type=Int,
                   default=lambda: Int(120),
                   required=False)
        spec.input("max_nodes",
                   valid_type=Int,
                   default=lambda: Int(24),
                   required=False)
        spec.input("mem_node",
                   valid_type=Int,
                   default=lambda: Int(64),
                   required=False)
        spec.input("pw_code", valid_type=Code)
        spec.input("pp_code", valid_type=Code)
        spec.input("projwfc_code", valid_type=Code)
        spec.input("structure", valid_type=StructureData)
        spec.input("precision",
                   valid_type=Float,
                   default=lambda: Float(1.0),
                   required=False)
        spec.input("num_export_bands",
                   valid_type=Int,
                   default=lambda: Int(8),
                   required=False)
        spec.input(
            'pseudo_family',
            valid_type=Str,
            required=True,
            help=
            'An alternative to specifying the pseudo potentials manually in `pseudos`: one can specify the name '
            'of an existing pseudo potential family and the work chain will generate the pseudos automatically '
            'based on the input structure.')
        # TODO: check why it does not work
        #spec.inputs("metadata.label", valid_type=six.string_types,
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
            while_(cls.should_run_export_orbitals)(cls.run_export_orbitals, ),
            cls.run_export_spinden,
            cls.run_closing,
        )
        spec.outputs.dynamic = True

        spec.exit_code(300, 'CALC_FAILED', message='The calculation failed.')

    # =========================================================================
    def run_cell_opt1(self):
        if self.inputs.optimize_cell.value:
            structure = self.inputs.structure
            return self._submit_pw_calc(structure,
                                        label="cell_opt1",
                                        runtype='vc-relax',
                                        precision=0.5,
                                        min_kpoints=int(1),
                                        max_nodes=self.inputs.max_nodes.value,
                                        mem_node=self.inputs.mem_node.value)
        self.report("Skipping: cell_opt = False")
        return

    # =========================================================================
    def run_cell_opt2(self):
        if self.inputs.optimize_cell.value:
            prev_calc = self.ctx.cell_opt1
            # ---
            # check if previous calc was okay
            error_msg = self._check_prev_calc(prev_calc)
            if error_msg is not None:
                return self.exit_codes.CALC_FAILED
            # ---
            structure = prev_calc.outputs.output_structure
            return self._submit_pw_calc(structure,
                                        label="cell_opt2",
                                        runtype='vc-relax',
                                        precision=1.0,
                                        min_kpoints=int(1),
                                        max_nodes=self.inputs.max_nodes.value,
                                        mem_node=self.inputs.mem_node.value)
        self.report("Skipping: cell_opt = False")
        return

    # =========================================================================
    def run_scf(self):
        if self.inputs.optimize_cell.value:
            prev_calc = self.ctx.cell_opt2
            # ---
            # check if previous calc was okay
            error_msg = self._check_prev_calc(prev_calc)
            if error_msg is not None:
                return self.exit_codes.CALC_FAILED
        # ---
            structure = prev_calc.outputs.output_structure
        else:
            structure = self.inputs.structure
        min_kpoints = min(int(10), self.inputs.max_kpoints.value)
        return self._submit_pw_calc(structure,
                                    label="scf",
                                    runtype='scf',
                                    precision=3.0,
                                    min_kpoints=min_kpoints,
                                    max_nodes=self.inputs.max_nodes.value,
                                    mem_node=self.inputs.mem_node.value,
                                    wallhours=4)

    # =========================================================================
    def run_export_hartree(self):
        self.report("Running pp.x to export hartree potential")
        label = "export_hartree"

        builder = PpCalculation.get_builder()
        builder.code = self.inputs.pp_code

        prev_calc = self.ctx.scf
        # ---
        # check if previous calc was okay
        error_msg = self._check_prev_calc(prev_calc)
        if error_msg is not None:
            return self.exit_codes.CALC_FAILED
        # ---
        builder.parent_folder = prev_calc.outputs.remote_folder

        structure = prev_calc.inputs.structure
        cell_a = structure.cell[0][0]
        cell_b = structure.cell[1][1]
        cell_c = structure.cell[2][2]

        builder.parameters = Dict(
            dict={
                'INPUTPP': {
                    'plot_num': 11,  # the V_bare + V_H potential
                },
                'PLOT': {
                    'iflag': 2,
                    'x0(1)': 0.0,
                    'x0(2)': 0.0,
                    'x0(3)': cell_c / cell_a,
                    # 3D vectors which determine the plotting plane
                    # in alat units)
                    'e1(1)': cell_a / cell_a,
                    'e1(2)': 0.0,
                    'e1(3)': 0.0,
                    'e2(1)': 0.0,
                    'e2(2)': cell_b / cell_a,
                    'e2(3)': 0.0,
                    'nx': 10,  # Number of points in the plane
                    'ny': 10,
                },
            })

        natoms = len(prev_calc.inputs.structure.attributes['sites'])
        nnodes = min(self.inputs.max_nodes.value,
                     (1 + int(natoms / self.inputs.mem_node.value)))
        # Reconsider the following lines, when https://gitlab.com/QEF/q-e/-/issues/221 is fixed.
        npools = 1
        #nnodes = int(prev_calc.attributes['resources']['num_machines'])
        #npools = int(prev_calc.inputs.settings.get_dict()['cmdline'][1])
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
        builder.settings = Dict(dict={'cmdline': ["-npools", str(npools)]})

        #        builder.metadata.options = {
        #            "resources": {
        #                "num_machines": int(1),
        #            },
        #            "max_wallclock_seconds": 1200,
        #            "withmpi": True,
        #        }

        running = self.submit(builder)
        return ToContext(**{label: running})

    # =========================================================================
    def run_bands(self):
        # ---
        # check if previous calc was okay
        error_msg = self._check_prev_calc(self.ctx.export_hartree)
        if error_msg is not None:
            return self.exit_codes.CALC_FAILED
        # ---
        prev_calc = self.ctx.scf
        structure = prev_calc.inputs.structure
        parent_folder = prev_calc.outputs.remote_folder
        min_kpoints = min(int(20), self.inputs.max_kpoints.value)
        return self._submit_pw_calc(structure,
                                    label="bands",
                                    parent_folder=parent_folder,
                                    runtype='bands',
                                    precision=4.0,
                                    min_kpoints=min_kpoints,
                                    max_nodes=self.inputs.max_nodes.value,
                                    mem_node=self.inputs.mem_node.value,
                                    wallhours=6)

    # =========================================================================
    def run_export_pdos(self):
        self.report("Running projwfc.x to export PDOS")
        label = "export_pdos"
        # ---
        # check if previous calc was okay
        error_msg = self._check_prev_calc(self.ctx.bands)
        if error_msg is not None:
            return self.exit_codes.CALC_FAILED
        # ---
        builder = ProjwfcCalculation.get_builder()
        builder.code = self.inputs.projwfc_code
        prev_calc = self.ctx.bands
        self._check_prev_calc(prev_calc)

        natoms = len(prev_calc.inputs.structure.attributes['sites'])
        nproc_mach = min(
            4, builder.code.computer.get_default_mpiprocs_per_machine())

        previous_nodes = int(prev_calc.attributes['resources']['num_machines'])
        previous_pools = int(
            prev_calc.inputs.settings.get_dict()['cmdline'][1])
        if natoms < 60:
            nnodes = min(int(2), previous_nodes)
            npools = min(int(2), previous_pools)
        elif natoms < int(120):
            nnodes = min(int(4), previous_nodes)
            npools = min(int(4), previous_pools)
        else:
            nnodes = previous_nodes
            npools = previous_pools
            nproc_mach = builder.code.computer.get_default_mpiprocs_per_machine(
            )

        nhours = 24
        builder.parent_folder = prev_calc.outputs.remote_folder

        # use the same number of pools as in bands calculation
        builder.parameters = Dict(
            dict={
                'projwfc': {
                    'ngauss': 1,
                    'degauss': 0.007,
                    'DeltaE': 0.01,
                    'filproj': 'projection.out',
                },
            })

        builder.metadata.label = label
        builder.metadata.options = {
            "resources": {
                "num_machines": int(nnodes),
                "num_mpiprocs_per_machine": nproc_mach,
            },
            "max_wallclock_seconds": nhours * 60 * 60,  # hours
            "withmpi": True,
        }

        builder.settings = Dict(
            dict={
                'additional_retrieve_list':
                ['./out/aiida.save/*.xml', '*_up', '*_down', '*_tot'],
                'cmdline': ["-npools", str(npools)],
            })

        future = self.submit(builder)
        return ToContext(**{label: future})

    # =========================================================================
    def run_bands_lowres(self):
        self.report("Running bands with fewer kpt to export KS")
        # ---
        # check if previous calc was okay
        error_msg = self._check_prev_calc(self.ctx.export_pdos)
        if error_msg is not None:
            return self.exit_codes.CALC_FAILED
        # ---
        prev_calc = self.ctx.scf
        structure = prev_calc.inputs.structure
        parent_folder = prev_calc.outputs.remote_folder
        min_kpoints = min(int(12), self.inputs.max_kpoints.value)
        return self._submit_pw_calc(structure,
                                    label="bands_lowres",
                                    parent_folder=parent_folder,
                                    runtype='bands',
                                    precision=0.0,
                                    min_kpoints=min_kpoints,
                                    max_nodes=self.inputs.max_nodes.value,
                                    mem_node=self.inputs.mem_node.value,
                                    wallhours=2)

    # =========================================================================
    def prepare_export_orbitals(self):
        self.report("Getting ready to export KS orbitals.")

        # Getting and checking the previous calculation.
        prev_calc = self.ctx.bands_lowres
        # ---
        # check if previous calc was okay
        error_msg = self._check_prev_calc(prev_calc)
        if error_msg is not None:
            return self.exit_codes.CALC_FAILED
        # ---
        self.ctx.export_orbitals_parameters = {
            'INPUTPP': {
                # contribution of a selected wavefunction
                # to charge density
                'plot_num':
                7,
                'kpoint(1)':
                int(1),
                'kpoint(2)':
                int(prev_calc.res.number_of_k_points *
                    prev_calc.res.number_of_spin_components),
                'kband(1)':
                0,
                'kband(2)':
                0,
            },
            'PLOT': {
                'iflag': 3,  # 3D plot
            },
        }

        nhours = int(2 + min(22, 2 * int(prev_calc.res.volume / 1500)))
        # Reconsider the following lines, when https://gitlab.com/QEF/q-e/-/issues/221 is fixed.
        natoms = len(prev_calc.inputs.structure.attributes['sites'])
        nnodes = min(self.inputs.max_nodes.value,
                     (1 + int(natoms / self.inputs.mem_node.value)))
        npools = 1
        self.ctx.export_orbitals_options = {
            "resources": {
                "num_machines":
                int(nnodes),
                #int(prev_calc.attributes['resources']['num_machines']),
                "num_mpiprocs_per_machine":
                self.inputs.pp_code.computer.get_default_mpiprocs_per_machine(
                ),
            },
            "max_wallclock_seconds": nhours * 60 * 60,  # 6 hours
            # Add the post-processing python scripts
            "withmpi": True,
            "parser_name": "nanotech_empa.pp",
        }
        #npools = int(prev_calc.inputs.settings['cmdline'][1])
        self.ctx.export_orbitals_settings = Dict(
            dict={'cmdline': ["-npools", str(npools)]})

        kband1 = max(
            int(prev_calc.res.number_of_electrons / 2) -
            int(self.inputs.num_export_bands.value / 2) + 1, 1)
        self.ctx.first_band = kband1
        self.ctx.export_orbitals_band_number = kband1

    def should_run_export_orbitals(self):
        prev_calc = self.ctx.bands_lowres
        kband2 = min(
            int(prev_calc.res.number_of_electrons / 2) +
            int(self.inputs.num_export_bands.value / 2),
            int(prev_calc.res.number_of_bands))
        return self.ctx.export_orbitals_band_number <= kband2

    def run_export_orbitals(self):
        self.report("Running pp.x to export KS orbitals")
        # ---
        # check if previous calc was okay
        if self.ctx.export_orbitals_band_number == self.ctx.first_band:
            to_check = 'bands_lowres'
        else:
            to_check = 'export_orbitals_{}'.format(
                self.ctx.export_orbitals_band_number - 1)

        error_msg = self._check_prev_calc(getattr(self.ctx, to_check))
        if error_msg is not None:
            return self.exit_codes.CALC_FAILED
        # ---
        builder = PpCalculation.get_builder()
        builder.code = self.inputs.pp_code
        prev_calc = self.ctx.bands_lowres
        builder.parent_folder = prev_calc.outputs.remote_folder
        builder.metadata.label = "export_orbitals"
        builder.metadata.options = self.ctx.export_orbitals_options
        builder.settings = self.ctx.export_orbitals_settings

        # Modifying the band number.
        self.ctx.export_orbitals_parameters['INPUTPP'][
            'kband(1)'] = self.ctx.export_orbitals_band_number
        self.ctx.export_orbitals_parameters['INPUTPP'][
            'kband(2)'] = self.ctx.export_orbitals_band_number
        builder.parameters = Dict(dict=self.ctx.export_orbitals_parameters)

        # Running the calculation.
        running = self.submit(builder)
        label = 'export_orbitals_{}'.format(
            self.ctx.export_orbitals_band_number)
        self.ctx.export_orbitals_band_number += 1
        return ToContext(**{label: running})

    # =========================================================================
    def run_export_spinden(self):
        self.report("Running pp.x to compute spinden")
        label = "export_spinden"
        last_ks = 'export_orbitals_{}'.format(
            self.ctx.export_orbitals_band_number - 1)
        # ---
        # check if previous calc was okay
        error_msg = self._check_prev_calc(getattr(self.ctx, last_ks))
        if error_msg is not None:
            return self.exit_codes.CALC_FAILED
        # ---
        builder = PpCalculation.get_builder()
        builder.code = self.inputs.pp_code
        nproc_mach = builder.code.computer.get_default_mpiprocs_per_machine()
        prev_calc = self.ctx.scf
        builder.parent_folder = prev_calc.outputs.remote_folder

        nspin = prev_calc.res.number_of_spin_components
        natoms = len(prev_calc.inputs.structure.attributes['sites'])
        nnodes = min(self.inputs.max_nodes.value,
                     (1 + int(natoms / self.inputs.mem_node.value)))
        # Reconsider the following lines, when https://gitlab.com/QEF/q-e/-/issues/221 is fixed.
        npools = 1
        #nnodes = int(prev_calc.attributes['resources']['num_machines'])
        #npools = int(prev_calc.inputs.settings.get_dict()['cmdline'][1])
        if nspin == 1:
            self.report("Skipping, got only one spin channel")
            return

        builder.parameters = Dict(
            dict={
                'INPUTPP': {
                    'plot_num': 6,  # spin polarization (rho(up)-rho(down))
                },
                'PLOT': {
                    'iflag': 3,  # 3D plot
                },
            })

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

        builder.settings = Dict(dict={'cmdline': ["-npools", str(npools)]})

        future = self.submit(builder)
        return ToContext(**{label: future})

    # =========================================================================

    def run_closing(self):
        self.report("Running final check")
        # Getting and checking the previous calculation.
        nspin = self.ctx.scf.res.number_of_spin_components

        # ---
        # check if previous calc was okay
        if nspin > 1:
            prev_calc = self.ctx.export_spinden
            error_msg = self._check_prev_calc(prev_calc)
            if error_msg is not None:
                return self.exit_codes.CALC_FAILED

            self.out('spin_density_arraydata',
                     self.ctx.export_spinden.outputs.output_data)

        self.report("END of workchain")
        return

    def _check_prev_calc(self, prev_calc):
        #error = None
        #output_fname = prev_calc.attributes['output_filename']
        if not prev_calc.is_finished_ok:
            if prev_calc.exit_status >= 500:
                self.report("Warning: previous step: " +
                            prev_calc.exit_message)
            else:
                self.report("ERROR: previous step: " + prev_calc.exit_message)
                return self.exit_codes.CALC_FAILED

    # =========================================================================
    def _submit_pw_calc(  # pylint: disable=too-many-arguments
            self,
            structure,
            label,
            runtype,
            precision,
            min_kpoints,
            max_nodes,
            mem_node,
            wallhours=24,
            parent_folder=None):
        self.report("Running pw.x for " + label)
        builder = PwCalculation.get_builder()

        builder.code = self.inputs.pw_code
        builder.structure = structure
        builder.parameters = self._get_parameters(structure, runtype, label)
        builder.pseudos = validate_and_prepare_pseudos_inputs(
            structure, None, self.inputs.pseudo_family)

        if parent_folder:
            builder.parent_folder = parent_folder

        # kpoints
        cell_a = builder.structure.cell[0][0]
        precision *= self.inputs.precision.value
        nkpoints = max(min_kpoints, int(30 * 2.5 / cell_a * precision))

        if self.inputs.max_kpoints.value < nkpoints:
            self.report("max kpoints exceeded, instead of  " + str(nkpoints) +
                        " k-points, using " + str(min_kpoints))
            nkpoints = self.inputs.max_kpoints.value  ## for test runs minimal memory

        use_symmetry = runtype != "bands"
        kpoints = self._get_kpoints(nkpoints, use_symmetry=use_symmetry)
        builder.kpoints = kpoints

        # parallelization settings
        ## TEMPORARY double pools in case of spin
        spinpools = int(1)
        start_mag = self._get_magnetization(structure)
        if any((m != 0 for m in start_mag.values())):
            spinpools = int(2)

        natoms = len(structure.sites)
        #        npools = spinpools * min(
        #            1 + int(nkpoints * 2.4 /
        #                    builder.code.computer.get_default_mpiprocs_per_machine()),
        #            int(5))
        max_npools = spinpools * min(1 + int(nkpoints / 4), int(6))
        nnodes_base = min(max_nodes, (1 + int(natoms / mem_node)))
        #nnodes = (1 + int(
        #    natoms * 0.2 /
        #    builder.code.computer.get_default_mpiprocs_per_machine())) * npools
        #nnodes = (1 + int(natoms / 60)) * npools

        guess_nnodes = max_npools * nnodes_base
        if guess_nnodes <= max_nodes:
            nnodes = guess_nnodes
            npools = max_npools
        else:
            nnodes = max_nodes
            npools = 1
            cpus_per_node = builder.code.computer.get_default_mpiprocs_per_machine(
            )
            for i in range(max_npools):
                if nnodes * cpus_per_node % (i + 1) == 0:
                    npools = i + 1
        if nnodes == 1 and builder.code.computer.get_default_mpiprocs_per_machine(
        ) == 1:
            npools = 1
        builder.metadata.label = label
        #nnodes.store
        builder.metadata.options = {
            "resources": {
                "num_machines": int(nnodes)
            },
            "withmpi": True,
            "max_wallclock_seconds": wallhours * 60 * 60,
        }

        builder.settings = Dict(dict={'cmdline': ["-npools", str(npools)]})

        future = self.submit(builder)
        return ToContext(**{label: future})

    # =========================================================================
    def _get_parameters(self, structure, runtype, label):
        params = {
            'CONTROL': {
                'calculation': runtype,
                'wf_collect': True,
                'forc_conv_thr': 0.0001,
                'nstep': 500,
            },
            'SYSTEM': {
                'ecutwfc': 50.,
                'ecutrho': 400.,
                'occupations': 'smearing',
                'degauss': 0.001,
            },
            'ELECTRONS': {
                'conv_thr': 1.e-8,
                'mixing_beta': 0.25,
                'electron_maxstep': 50,
                'scf_must_converge': False,
            },
        }

        if label == 'cell_opt1':
            params['CONTROL']['forc_conv_thr'] = 0.0005
        if runtype == "vc-relax":
            # in y and z direction there is only vacuum
            params['CELL'] = {'cell_dofree': 'x'}

        # if runtype == "bands":
        #     params['CONTROL']['restart_mode'] = 'restart'

        start_mag = self._get_magnetization(structure)
        if any((m != 0 for m in start_mag.values())):
            params['SYSTEM']['nspin'] = 2
            params['SYSTEM']['starting_magnetization'] = start_mag

        return Dict(dict=params)

    # =========================================================================
    def _get_kpoints(self, nx, use_symmetry=True):
        nx = max(1, nx)

        kpoints = KpointsData()
        if use_symmetry:
            kpoints.set_kpoints_mesh([nx, 1, 1], offset=[0.0, 0.0, 0.0])
        else:
            # List kpoints explicitly.
            points = [[r, 0.0, 0.0] for r in np.linspace(0, 0.5, nx)]
            kpoints.set_kpoints(points)
        return kpoints

    # =========================================================================
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
