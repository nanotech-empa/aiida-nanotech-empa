from aiida_nanotech_empa.workflows.gaussian import common
from aiida_nanotech_empa.utils import common_utils

from aiida.engine import WorkChain, ToContext, ExitCode, while_, if_
from aiida.orm import Int, Str, Code, Dict, Bool, List, StructureData, RemoteData

from aiida.plugins import WorkflowFactory

GaussianBaseWorkChain = WorkflowFactory('gaussian.base')
GaussianCubesWorkChain = WorkflowFactory('gaussian.cubes')

GaussianCasscfWorkChain = WorkflowFactory('nanotech_empa.gaussian.casscf')

GaussianHfMp2WorkChain = WorkflowFactory('nanotech_empa.gaussian.hf_mp2')


class GaussianCasscfSeriesWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input('gaussian_code', valid_type=Code)

        spec.input('nm_list',
                   valid_type=List,
                   required=True,
                   help='Successive list of (n,m) tuples to run CAS(n,m).')

        spec.input('structure',
                   valid_type=StructureData,
                   required=True,
                   help='input geometry')

        spec.input('init_functional',
                   valid_type=Str,
                   required=False,
                   default=lambda: Str('UHF'),
                   help='Functional for the initial orbitals.')

        spec.input(
            "start_calc_folder",
            valid_type=RemoteData,
            required=False,
            help="Read starting orbitals from here instead.",
        )

        spec.input('basis_set',
                   valid_type=Str,
                   required=True,
                   help='Basis set')

        spec.input('multiplicity_list',
                   valid_type=List,
                   required=False,
                   default=lambda: List(list=[1, 3]),
                   help='spin multiplicity')

        spec.input("start_uno",
                   valid_type=Bool,
                   required=False,
                   default=lambda: Bool(True),
                   help='Use natural orbitals of the start calculation.')

        spec.input("mp2",
                   valid_type=Bool,
                   required=False,
                   default=lambda: Bool(False),
                   help='calculate the MP2 correction (CASMP2).')

        spec.input("num_orbital_cubes",
                   valid_type=Int,
                   required=False,
                   default=lambda: Int(0),
                   help='Generate cubes for orbitals (n*occ and n*virt).')

        spec.input("formchk_code", valid_type=Code, required=False)
        spec.input("cubegen_code", valid_type=Code, required=False)

        spec.input(
            "options",
            valid_type=Dict,
            required=False,
            help="Use custom metadata.options instead of the automatic ones.",
        )

        spec.outline(
            cls.setup,
            if_(cls.should_do_init)(cls.initial_scf),
            while_(cls.any_multiplicity_left)(while_(cls.any_casscf_nm_left)(
                cls.casscf)), cls.finalize)

        spec.outputs.dynamic = True

        spec.exit_code(
            302,
            "ERROR_OPTIONS",
            message="Input options are invalid.",
        )
        spec.exit_code(
            390,
            "ERROR_TERMINATION",
            message="One or more steps of the work chain failed.",
        )

    def setup(self):
        self.report("Inspecting input and setting up things")

        pymatgen_structure = self.inputs.structure.get_pymatgen_molecule()
        self.ctx.n_atoms = pymatgen_structure.num_sites
        self.ctx.n_electrons = pymatgen_structure.nelectrons

        self.ctx.comp = self.inputs.gaussian_code.computer

        success = common.determine_metadata_options(self)
        if not success:
            return self.exit_codes.ERROR_OPTIONS

        num_cores, memory_mb = common.get_gaussian_cores_and_memory(
            self.ctx.metadata_options, self.ctx.comp)

        self.ctx.link0 = {
            '%chk': 'aiida.chk',
            '%mem': "%dMB" % memory_mb,
            '%nprocshared': str(num_cores),
        }

        self.ctx.i_current_mult = 0
        self.ctx.i_current_nm = 0

        self.ctx.last_submitted_label = None

        return ExitCode(0)

    def should_do_init(self):
        return 'start_calc_folder' not in self.inputs

    def initial_scf(self):

        self.report("Submitting initial SCF")

        self.ctx.init_mult = list(self.inputs.multiplicity_list)[0]

        parameters = Dict(
            dict={
                'link0_parameters': self.ctx.link0.copy(),
                'dieze_tag': '#P',
                'functional': self.inputs.init_functional.value,
                'basis_set': self.inputs.basis_set.value,
                'charge': 0,
                'multiplicity': self.ctx.init_mult,
                'route_parameters': {
                    'scf': {
                        'maxcycle': 128,
                    },
                    'stable': 'opt',
                },
            })

        if self.ctx.init_mult == 1:
            parameters['route_parameters']['guess'] = "mix"

        builder = GaussianBaseWorkChain.get_builder()
        builder.gaussian.parameters = parameters
        builder.gaussian.structure = self.inputs.structure
        builder.gaussian.code = self.inputs.gaussian_code
        builder.gaussian.metadata.options = self.ctx.metadata_options

        future = self.submit(builder)

        self.ctx.last_submitted_label = 'init_scf'
        return ToContext(init_scf=future)

    def any_multiplicity_left(self):
        if self.ctx.i_current_mult == len(self.inputs.multiplicity_list):
            return False
        self.ctx.current_mult = list(
            self.inputs.multiplicity_list)[self.ctx.i_current_mult]
        self.ctx.i_current_mult += 1
        self.ctx.i_current_nm = 0
        return True

    def any_casscf_nm_left(self):
        if self.ctx.i_current_nm == len(self.inputs.nm_list):
            return False
        self.ctx.current_nm = list(self.inputs.nm_list)[self.ctx.i_current_nm]
        self.ctx.i_current_nm += 1
        return True

    def casscf(self):

        if self.ctx.last_submitted_label is not None:
            if not common_utils.check_if_calc_ok(
                    self, self.ctx[self.ctx.last_submitted_label]):
                return self.exit_codes.ERROR_TERMINATION

        # Set output already for the previous step
        #lsc = self.ctx.last_submitted_label
        #if lsc.startswith('cas'):
        #    self.out(f"{lsc}_out_params",
        #             self.ctx[lsc].outputs.output_parameters)
        #    if 'cube_image_folder' in self.ctx[lsc].outputs:
        #        self.out(f"{lsc}_cube_image_folder",
        #                 self.ctx[lsc].outputs.cube_image_folder)

        uno = False

        # determine previous node
        if self.ctx.i_current_mult == 1 and self.ctx.i_current_nm == 1:
            # first one uses the initial_scf or the specified calculation
            if 'start_calc_folder' in self.inputs:
                prev_calc_folder = self.inputs.start_calc_folder
            else:
                prev_calc_folder = self.ctx.init_scf.outputs.remote_folder
            if self.inputs.start_uno:
                uno = True
        elif self.ctx.i_current_mult == 1:
            # for the "base" multiplicity, use the natural orbitals of the previous casscf
            prev_nm = list(self.inputs.nm_list)[self.ctx.i_current_nm - 2]
            prev_label = f"cas_{prev_nm[0]}_{prev_nm[1]}_m{self.ctx.current_mult}"
            prev_calc_folder = self.ctx[prev_label].outputs.remote_folder
        else:
            # for any other multiplicity, use the orbitals of the corresponding base casscf
            prev_label = "cas_{}_{}_m{}".format(
                self.ctx.current_nm[0], self.ctx.current_nm[1],
                self.inputs.multiplicity_list[0])
            prev_calc_folder = self.ctx[prev_label].outputs.remote_folder

        builder = GaussianCasscfWorkChain.get_builder()

        builder.parent_calc_folder = prev_calc_folder

        builder.gaussian_code = self.inputs.gaussian_code

        builder.n = Int(self.ctx.current_nm[0])
        builder.m = Int(self.ctx.current_nm[1])

        builder.basis_set = self.inputs.basis_set

        builder.multiplicity = Int(self.ctx.current_mult)

        builder.uno = Bool(uno)
        builder.mp2 = self.inputs.mp2

        codes_set = 'formchk_code' in self.inputs and 'cubegen_code' in self.inputs
        if self.inputs.num_orbital_cubes > 0 and codes_set:
            builder.num_orbital_cubes = self.inputs.num_orbital_cubes
            builder.formchk_code = self.inputs.formchk_code
            builder.cubegen_code = self.inputs.cubegen_code

        builder.options = self.inputs.options

        label = "cas_{}_{}_m{}".format(self.ctx.current_nm[0],
                                       self.ctx.current_nm[1],
                                       self.ctx.current_mult)

        submitted_node = self.submit(builder)

        self.ctx.last_submitted_label = label
        return ToContext(**{label: submitted_node})

    def finalize(self):

        self.report("Finalizing...")

        if not common_utils.check_if_calc_ok(
                self, self.ctx[self.ctx.last_submitted_label]):
            return self.exit_codes.ERROR_TERMINATION

        for var in self.ctx:
            if var.startswith('cas'):
                self.out(f"{var}_out_params",
                         self.ctx[var].outputs.output_parameters)
                if 'cube_image_folder' in self.ctx[var].outputs:
                    self.out(f"{var}_cube_image_folder",
                             self.ctx[var].outputs.cube_image_folder)

        return ExitCode(0)
