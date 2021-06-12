from aiida.engine.processes.functions import calcfunction
from aiida_nanotech_empa.workflows.gaussian import common
from aiida_nanotech_empa.utils import common_utils

from aiida.engine import WorkChain, ToContext, if_, ExitCode
from aiida.orm import Int, Str, Code, Dict, Bool, RemoteData, List

from aiida.plugins import WorkflowFactory

GaussianBaseWorkChain = WorkflowFactory('gaussian.base')
GaussianCubesWorkChain = WorkflowFactory('gaussian.cubes')


@calcfunction
def add_mp2_to_out_params(out_params, mp2_energy):
    new_params = dict(out_params)
    new_params['casmp2_energy_ev'] = mp2_energy.value
    return Dict(dict=new_params)


class GaussianCasscfWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input('gaussian_code', valid_type=Code)

        spec.input(
            "parent_calc_folder",
            valid_type=RemoteData,
            required=True,
            help="parent Gaussian calculation directory",
        )

        spec.input('n',
                   valid_type=Int,
                   required=True,
                   help='Number of electrons CAS(n,m).')

        spec.input('m',
                   valid_type=Int,
                   required=True,
                   help='Number of orbitals CAS(n,m).')

        spec.input('basis_set',
                   valid_type=Str,
                   required=True,
                   help='basis_set')

        spec.input('multiplicity',
                   valid_type=Int,
                   required=False,
                   default=lambda: Int(1),
                   help='spin multiplicity')

        spec.input(
            "uno",
            valid_type=Bool,
            required=False,
            default=lambda: Bool(False),
            help='Use the natural orbitals from the previous calculation.')

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

        spec.outline(cls.setup, cls.casscf,
                     if_(cls.should_do_mp2)(cls.casmp2),
                     if_(cls.should_do_cubes)(cls.cubes), cls.finalize)

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

    def should_do_mp2(self):
        return self.inputs.mp2

    def should_do_cubes(self):
        codes_set = 'formchk_code' in self.inputs and 'cubegen_code' in self.inputs
        return codes_set and self.inputs.num_orbital_cubes.value > 0

    def setup(self):
        self.report("Inspecting input and setting up things")

        self.ctx.comp = self.inputs.gaussian_code.computer

        if 'options' in self.inputs:
            val = common.validate_metadata_options(dict(self.inputs.options),
                                                   self.ctx.comp)
            if val is not None:
                self.report("Error: " + val)
                return self.exit_codes.ERROR_OPTIONS
            self.ctx.metadata_options = dict(self.inputs.options)

        else:
            self.ctx.metadata_options = common.get_default_metadata_options(
                200, self.ctx.comp, self.inputs.basis_set.value)

        num_cores, memory_mb = common.get_gaussian_cores_and_memory(
            self.ctx.metadata_options, self.ctx.comp)

        self.ctx.link0 = {
            '%chk': 'aiida.chk',
            '%mem': "%dMB" % memory_mb,
            '%nprocshared': str(num_cores),
            '%oldchk': 'parent_calc/aiida.chk',
        }

        return ExitCode(0)

    def casscf(self):

        self.report("Submitting CASSCF")

        func_str = 'CASSCF({},{}{})'.format(self.inputs.n.value,
                                            self.inputs.m.value,
                                            ",UNO" if self.inputs.uno else "")

        parameters = Dict(
            dict={
                'link0_parameters': self.ctx.link0.copy(),
                'dieze_tag': '#P',
                'functional': func_str,
                'basis_set': self.inputs.basis_set.value,
                'charge': 0,
                'multiplicity': self.inputs.multiplicity.value,
                'route_parameters': {
                    'scf': {
                        'maxcycle': 512,
                    },
                    'geom': 'checkpoint',
                    'guess': 'read',
                    'pop': 'naturalorbital',
                    'sp': None
                },
            })

        builder = GaussianBaseWorkChain.get_builder()
        builder.gaussian.parameters = parameters
        builder.gaussian.code = self.inputs.gaussian_code
        builder.gaussian.parent_calc_folder = self.inputs.parent_calc_folder
        builder.gaussian.metadata.options = self.ctx.metadata_options
        builder.gaussian.metadata.options[
            'parser_name'] = 'nanotech_empa.gaussian.casscf'

        future = self.submit(builder)
        return ToContext(casscf=future)

    def casmp2(self):

        if not common_utils.check_if_calc_ok(self, self.ctx.casscf):
            return self.exit_codes.ERROR_TERMINATION

        self.report("Submitting CASMP2")

        func_str = 'CASSCF({},{})'.format(self.inputs.n.value,
                                          self.inputs.m.value)

        parameters = Dict(
            dict={
                'link0_parameters': self.ctx.link0.copy(),
                'dieze_tag': '#P',
                'functional': func_str,
                'basis_set': self.inputs.basis_set.value,
                'charge': 0,
                'multiplicity': self.inputs.multiplicity.value,
                'route_parameters': {
                    'scf': {
                        'maxcycle': 512,
                    },
                    'geom': 'checkpoint',
                    'guess': 'read',
                    'sp': None,
                    'mp2': None,
                },
            })

        builder = GaussianBaseWorkChain.get_builder()
        builder.gaussian.parameters = parameters
        builder.gaussian.code = self.inputs.gaussian_code
        builder.gaussian.parent_calc_folder = self.ctx.casscf.outputs.remote_folder
        builder.gaussian.metadata.options = self.ctx.metadata_options
        builder.gaussian.metadata.options[
            'parser_name'] = 'nanotech_empa.gaussian.casscf'

        future = self.submit(builder)
        return ToContext(casmp2=future)

    def cubes(self):

        n_d = self.inputs.num_orbital_cubes.value
        n_u = self.inputs.num_orbital_cubes.value

        builder = GaussianCubesWorkChain.get_builder()
        builder.formchk_code = self.inputs.formchk_code
        builder.cubegen_code = self.inputs.cubegen_code
        builder.gaussian_calc_folder = self.ctx.casscf.outputs.remote_folder
        builder.gaussian_output_params = self.ctx.casscf.outputs.output_parameters
        builder.orbital_indexes = List(list=list(range(-n_d + 1, n_u + 1)))
        builder.natural_orbitals = Bool(True)

        builder.cubegen_parser_name = 'nanotech_empa.gaussian.cubegen_pymol'
        builder.cubegen_parser_params = Dict(dict={
            'isovalues': [0.050],
            'orient_cube': True
        })
        future = self.submit(builder)
        return ToContext(cubes=future)

    def finalize(self):

        self.report("Finalizing...")

        if not common_utils.check_if_calc_ok(self, self.ctx.casscf):
            return self.exit_codes.ERROR_TERMINATION

        out_params = self.ctx.casscf.outputs.output_parameters

        if self.should_do_mp2():
            if not common_utils.check_if_calc_ok(self, self.ctx.casmp2):
                return self.exit_codes.ERROR_TERMINATION
            self.out('casmp2_energy_ev',
                     self.ctx.casmp2.outputs.casmp2_energy_ev)
            out_params = add_mp2_to_out_params(
                out_params, self.ctx.casmp2.outputs.casmp2_energy_ev)

        if self.should_do_cubes():
            if not common_utils.check_if_calc_ok(self, self.ctx.cubes):
                return self.exit_codes.ERROR_TERMINATION
            self.out('cube_image_folder',
                     self.ctx.cubes.outputs.cube_image_folder)

        self.out('output_parameters', out_params)
        self.out('casscf_energy_ev', self.ctx.casscf.outputs.casscf_energy_ev)

        self.out('remote_folder', self.ctx.casscf.outputs.remote_folder)

        return ExitCode(0)
