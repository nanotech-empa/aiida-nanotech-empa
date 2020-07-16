import numpy as np

# AiiDA imports
from aiida.orm import Code, Dict, Int, Float, KpointsData, Str, StructureData, SinglefileData
from aiida.engine import WorkChain, ToContext, run, submit
#from aiida.orm.nodes.data.upf import get_pseudos_dict, get_pseudos_from_structure

# aiida_quantumespresso imports
from aiida.engine import ExitCode
from aiida_quantumespresso.calculations.pw import PwCalculation
from aiida_quantumespresso.calculations.pp import PpCalculation
from aiida_quantumespresso.calculations.projwfc import ProjwfcCalculation
from aiida_quantumespresso.utils.pseudopotential import validate_and_prepare_pseudos_inputs


class NanoribbonWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super(NanoribbonWorkChain, cls).define(spec)
        spec.input("pw_code", valid_type=Code)
        spec.input("pp_code", valid_type=Code)
        spec.input("projwfc_code", valid_type=Code)
        spec.input("structure", valid_type=StructureData)
        spec.input("precision",
                   valid_type=Float,
                   default=Float(1.0),
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
            cls.run_export_orbitals,
            cls.run_export_spinden,
        )
        spec.outputs.dynamic = True

    # =========================================================================
    def run_cell_opt1(self):
        structure = self.inputs.structure
        return self._submit_pw_calc(structure,
                                    label="cell_opt1",
                                    runtype='vc-relax',
                                    precision=0.5,
                                    min_kpoints=int(1))

    # =========================================================================
    def run_cell_opt2(self):
        prev_calc = self.ctx.cell_opt1
        self._check_prev_calc(prev_calc)
        structure = prev_calc.outputs.output_structure
        return self._submit_pw_calc(structure,
                                    label="cell_opt2",
                                    runtype='vc-relax',
                                    precision=1.0,
                                    min_kpoints=int(1))

    # =========================================================================
    def run_scf(self):
        prev_calc = self.ctx.cell_opt2
        self._check_prev_calc(prev_calc)
        structure = prev_calc.outputs.output_structure
        return self._submit_pw_calc(structure,
                                    label="scf",
                                    runtype='scf',
                                    precision=3.0,
                                    min_kpoints=int(10),
                                    wallhours=4)

    # =========================================================================
    def run_export_hartree(self):
        self.report("Running pp.x to export hartree potential")
        label = "export_hartree"

        builder = PpCalculation.get_builder()
        builder.code = self.inputs.pp_code

        prev_calc = self.ctx.scf
        self._check_prev_calc(prev_calc)
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

        builder.metadata.label = label
        builder.metadata.options = {
            "resources": {
                "num_machines": int(1),
            },
            "max_wallclock_seconds": 1200,
            "withmpi": True,
        }

        running = self.submit(builder)
        return ToContext(**{label: running})

    # =========================================================================
    def run_bands(self):
        prev_calc = self.ctx.scf
        self._check_prev_calc(prev_calc)
        structure = prev_calc.inputs.structure
        parent_folder = prev_calc.outputs.remote_folder
        return self._submit_pw_calc(structure,
                                    label="bands",
                                    parent_folder=parent_folder,
                                    runtype='bands',
                                    precision=4.0,
                                    min_kpoints=int(20),
                                    wallhours=6)

    # =========================================================================
    def run_bands_lowres(self):
        prev_calc = self.ctx.scf
        self._check_prev_calc(prev_calc)
        structure = prev_calc.inputs.structure
        parent_folder = prev_calc.outputs.remote_folder
        return self._submit_pw_calc(structure,
                                    label="bands_lowres",
                                    parent_folder=parent_folder,
                                    runtype='bands',
                                    precision=0.0,
                                    min_kpoints=int(12),
                                    wallhours=2)

    # =========================================================================
    def run_export_orbitals(self):
        self.report("Running pp.x to export KS orbitals")
        builder = PpCalculation.get_builder()
        builder.code = self.inputs.pp_code
        nproc_mach = builder.code.computer.get_default_mpiprocs_per_machine()

        prev_calc = self.ctx.bands_lowres
        self._check_prev_calc(prev_calc)
        builder.parent_folder = prev_calc.outputs.remote_folder

        nel = prev_calc.res.number_of_electrons
        nkpt = prev_calc.res.number_of_k_points
        nbnd = prev_calc.res.number_of_bands
        nspin = prev_calc.res.number_of_spin_components
        volume = prev_calc.res.volume
        kband1 = max(int(nel / 2) - int(1), int(1))
        kband2 = min(int(nel / 2) + int(2), int(nbnd))
        kpoint1 = int(1)
        kpoint2 = int(nkpt * nspin)
        nhours = int(2 + min(22, 2 * int(volume / 1500)))

        nnodes = int(prev_calc.attributes['resources']['num_machines'])
        npools = int(prev_calc.inputs.settings['cmdline'][1])
        for inb in range(kband1, kband2 + 1):
            builder.parameters = Dict(
                dict={
                    'INPUTPP': {
                        # contribution of a selected wavefunction
                        # to charge density
                        'plot_num': 7,
                        'kpoint(1)': kpoint1,
                        'kpoint(2)': kpoint2,
                        'kband(1)': inb,
                        'kband(2)': inb,
                    },
                    'PLOT': {
                        'iflag': 3,  # 3D plot
                    },
                })

            builder.metadata.label = "export_orbitals"
            builder.metadata.options = {
                "resources": {
                    "num_machines": nnodes,
                    "num_mpiprocs_per_machine": nproc_mach,
                },
                "max_wallclock_seconds": nhours * 60 * 60,  # 6 hours
                # Add the post-processing python scripts
                "withmpi": True,
                "parser_name": "nanotech_empa.pp",
            }

            builder.settings = Dict(dict={'cmdline': ["-npools", str(npools)]})
            running = self.submit(builder)
            label = 'export_orbitals_{}'.format(inb)
            self.to_context(**{label: running})

    # =========================================================================
    def run_export_spinden(self):
        self.report("Running pp.x to compute spinden")
        label = "export_spinden"

        builder = PpCalculation.get_builder()
        builder.code = self.inputs.pp_code
        nproc_mach = builder.code.computer.get_default_mpiprocs_per_machine()
        prev_calc = self.ctx.scf
        self._check_prev_calc(prev_calc)
        builder.parent_folder = prev_calc.outputs.remote_folder

        nspin = prev_calc.res.number_of_spin_components
        nnodes = int(prev_calc.attributes['resources']['num_machines'])
        npools = int(prev_calc.inputs.settings.get_dict()['cmdline'][1])
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
                "num_machines": nnodes,
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
    def run_export_pdos(self):
        self.report("Running projwfc.x to export PDOS")
        label = "export_pdos"

        builder = ProjwfcCalculation.get_builder()
        builder.code = self.inputs.projwfc_code
        prev_calc = self.ctx.bands
        self._check_prev_calc(prev_calc)

        natoms = len(prev_calc.inputs.structure.attributes['sites'])
        nproc_mach = 4

        if natoms < 60:
            nnodes = int(2)
            npools = int(2)
        elif natoms < int(120):
            nnodes = int(4)
            npools = int(4)
        else:
            nnodes = int(prev_calc.attributes['resources']['num_machines'])
            npools = int(prev_calc.inputs.settings.get_dict()['cmdline'][1])
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
                "num_machines": nnodes,
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
    def _check_prev_calc(self, prev_calc):
        error = None
        output_fname = prev_calc.attributes['output_filename']
        if not prev_calc.is_finished_ok:
            error = "Previous calculation failed"  #in state: "+prev_calc.get_state()
        elif output_fname not in prev_calc.outputs.retrieved.list_object_names(
        ):
            error = "Previous calculation did not retrive {}".format(
                output_fname)
        else:
            content = prev_calc.outputs.retrieved.get_object_content(
                output_fname)
            if "JOB DONE." not in content:
                error = "Previous calculation not DONE."
        if error:
            self.report("ERROR: " + error)
            return ExitCode(450)

    # =========================================================================
    def _submit_pw_calc(self,
                        structure,
                        label,
                        runtype,
                        precision,
                        min_kpoints,
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
        use_symmetry = runtype != "bands"
        kpoints = self._get_kpoints(nkpoints, use_symmetry=use_symmetry)
        builder.kpoints = kpoints

        # parallelization settings
        ## TEMPORARY double pools in case of spin
        spinpools = int(1)
        start_mag = self._get_magnetization(structure)
        if any([m != 0 for m in start_mag.values()]):
            spinpools = int(2)

        natoms = len(structure.sites)
        npools = spinpools * min(
            1 + int(nkpoints * 2.4 /
                    builder.code.computer.get_default_mpiprocs_per_machine()),
            int(5))
        nnodes = (1 + int(
            natoms * 0.2 /
            builder.code.computer.get_default_mpiprocs_per_machine())) * npools

        builder.metadata.label = label
        builder.metadata.options = {
            "resources": {
                "num_machines": nnodes
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
        if any([m != 0 for m in start_mag.values()]):
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
