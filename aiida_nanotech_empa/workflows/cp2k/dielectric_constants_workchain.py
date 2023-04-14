import pathlib
#import ruamel.yaml as yaml
import yaml
from copy import deepcopy
import numpy as np

from aiida.engine import ExitCode, ToContext, WorkChain

from aiida.plugins import DataFactory, WorkflowFactory
from aiida.common import AttributeDict
from aiida_nanotech_empa.workflows.cp2k.cp2k_utils import dict_merge

Cp2kBaseWorkChain = WorkflowFactory("cp2k.base")


Dict = DataFactory("core.dict")
StructureData = DataFactory("core.structure")
Str = DataFactory("core.str")

ALLOWED_PROTOCOLS = ["debug", "standard"]

#This method only works for systems with a bandgap and OT, for future (DFT+U)
def get_kinds_section(structure, protocol_settings):
    """ Write the &KIND sections given the structure and the settings_dict"""
    kinds = []
    all_atoms = set(structure.get_ase().get_chemical_symbols())
    for atom in all_atoms:
        kinds.append({
            '_': atom,
            'BASIS_SET': protocol_settings['basis_set'][atom],
            'POTENTIAL': protocol_settings['pseudopotential'][atom],
        })
    return {'FORCE_EVAL': {'SUBSYS': {'KIND': kinds}}}

def check_symmetry(cell_lengths, angles):
    symmetry = "NONE"
    super_cell = "1 1 1"
    a, b, c = cell_lengths
    alpha, beta, gamma = angles
    
    if a == b == c and alpha == beta == gamma == 90:
        symmetry = "CUBIC"
    elif a == b != c and alpha == beta == 90 and gamma == 120:
        symmetry = "HEXAGONAL"
    elif a != b != c and alpha == gamma == 90 and beta != 90:
        symmetry = "MONOCLINIC"
    elif a == b != c and alpha == beta == 90 and gamma != 90:
        symmetry = "MONOCLINIC_GAMMA_AB"
    elif a != b != c and alpha == beta == gamma == 90:
        symmetry = "ORTHORHOMBIC"
    elif a == b == c and alpha == beta == gamma and alpha != 90:
        symmetry = "RHOMBOHEDRAL"
    elif a == b != c and alpha == beta == gamma == 90:
        symmetry = "TETRAGONAL_AB"
    elif a == c != b and alpha == beta == gamma == 90:
        symmetry = "TETRAGONAL_AC"
    elif a != b == c and alpha == beta == gamma == 90:
        symmetry = "TETRAGONAL_BC"
    elif a != b != c and alpha != beta != gamma and alpha != 90 and beta != 90 and gamma != 90:
        symmetry = "TRICLINIC"

    value_expand= lambda variable: round(14/variable) if round(14/variable) != 0 else 1
    super_cell = "{i} {j} {k}".format(i=value_expand(a), j=value_expand(b), k=value_expand(c)) #14 Angstrom cutoff
    return symmetry, super_cell


class Cp2kDielectricWorkChain(WorkChain):
    """A workchain to compute the dielectric constanst using the formalism from P. Umari (Phys. Rev. Lett. 89, 157602)
    in cp2k.
    """
    @classmethod
    def define(cls, spec):
        """Specify input, outputs, and the workchain outline."""
        super().define(spec)
        spec.input('structure', valid_type=StructureData)
        spec.input('protocol',valid_type=Str,
            default=lambda: Str('standard'),required=False,help="Either 'standard', 'debug'",)
        spec.expose_inputs(Cp2kBaseWorkChain, namespace='cp2k_base', exclude=['cp2k.structure', 'cp2k.parameters'] )
        spec.outline(
            cls.setup,
            cls.run_geoopt,
            cls.run_efield0,
            cls.run_efieldxyz_geoopt,
            cls.results
        )
        spec.output('epsilon', valid_type=Dict, help='Dictionary with the dielectric constants')
        spec.exit_code(390,"ERROR_TERMINATION",
            message="One or more steps of the work chain failed.",)
        
    def setup(self):
        """Perform the initial setup."""
        self.ctx.structure = self.inputs.structure
        
        if self.inputs.protocol not in ALLOWED_PROTOCOLS:
            self.report("Error: protocol not supported.")
            return self.exit_codes.ERROR_TERMINATION

        #Get parameters for ground state
        path_settings = pathlib.Path(__file__).parent / "./protocols/efield_geo.yaml"
        with open(path_settings, 'r') as stream:
            self.ctx.protocols = yaml.safe_load(stream)
        if self.inputs.protocol == 'debug':
            parameters = deepcopy(self.ctx.protocols['debug']['settings'])
        else:
            parameters = deepcopy(self.ctx.protocols['standard']['settings'])
            cell_lengths = self.inputs.structure.cell_lengths
            cell_angles = self.inputs.structure.cell_angles
            symmetry, super_cell = check_symmetry(cell_lengths, cell_angles)
            symmetry_parameters = {
                'FORCE_EVAL': {
                    'SUBSYS': {
                        'CELL': {
                            'MULTIPLE_UNIT_CELL': super_cell,
                            'SYMMETRY': symmetry,
                        },
                        'TOPOLOGY': {
                            'MULTIPLE_UNIT_CELL': super_cell,
                        },
                    }
                }
            }
            dict_merge(parameters,symmetry_parameters)

        kinds = get_kinds_section(self.ctx.structure, self.ctx.protocols)
        dict_merge(parameters,kinds)
        self.ctx.parameters = parameters

    def run_geoopt(self):

        ground_state = AttributeDict(self.exposed_inputs(Cp2kBaseWorkChain, namespace='cp2k_base'))
        ground_state['cp2k']['structure'] = self.ctx.structure
        ground_state['cp2k']['parameters'] = Dict(dict=self.ctx.parameters)
        running_ground_state = self.submit(Cp2kBaseWorkChain, **ground_state)
        self.report(f'Submitting ground state cell optimization')
        self.to_context(ground_state=running_ground_state)

    def run_efield0(self):

        optimized_structure = self.ctx.ground_state.outputs.output_structure
        self.ctx.efield_parameters = self.ctx.parameters
        if 'CELL' in self.ctx.efield_parameters['FORCE_EVAL']['SUBSYS']:
            self.ctx.efield_parameters['FORCE_EVAL']['SUBSYS'].pop('CELL')
        if 'TOPOLOGY' in self.ctx.efield_parameters['FORCE_EVAL']['SUBSYS']:
            self.ctx.efield_parameters['FORCE_EVAL']['SUBSYS'].pop('TOPOLOGY')

        self.ctx.efield_parameters['GLOBAL']['RUN_TYPE'] = 'ENERGY'
        efield_parameters = {
            'FORCE_EVAL': {
                'DFT': {
                    'WFN_RESTART_FILE_NAME': './parent_calc/aiida-RESTART.wfn',
                    'SCF': {
                        'SCF_GUESS': 'RESTART',
                    },
                    'PERIODIC_EFIELD' : {
                        'INTENSITY': 0.000,
                        'POLARISATION': '1 0 0'  
                    },
                    'PRINT': {
                        'MOMENTS' : {
                            'PERIODIC': True
                        },
                        
                    },
                }
            }

        }
        dict_merge(self.ctx.efield_parameters,efield_parameters)
        efield0 = AttributeDict(self.exposed_inputs(Cp2kBaseWorkChain, namespace='cp2k_base'))
        efield0['cp2k']['structure'] = optimized_structure
        efield0['cp2k']['parameters'] = Dict(dict=self.ctx.efield_parameters)
        efield0['cp2k']['parent_calc_folder'] = self.ctx.ground_state.outputs.remote_folder
        running_efield0 = self.submit(Cp2kBaseWorkChain, **efield0)
        self.report(f'Submiting single point field 0')
        self.to_context(efield0=running_efield0)
    
    def run_efieldxyz_geoopt(self):
        direction = {'x': '1 0 0', 'y':'0 1 0', 'z': '0 0 1'}
        optimized_structure = self.ctx.ground_state.outputs.output_structure
        if 'STRESS_TENSOR' in self.ctx.efield_parameters['FORCE_EVAL']:
            self.ctx.efield_parameters['FORCE_EVAL'].pop('STRESS_TENSOR')
        self.ctx.efield_parameters['GLOBAL']['RUN_TYPE'] = 'GEO_OPT'
        self.ctx.efield_parameters['FORCE_EVAL']['DFT']['PERIODIC_EFIELD']['INTENSITY'] = 0.0005

        for vector in direction:
            geo_efield = AttributeDict(self.exposed_inputs(Cp2kBaseWorkChain, namespace='cp2k_base'))
            geo_efield['cp2k']['structure'] = optimized_structure
            self.ctx.efield_parameters['FORCE_EVAL']['DFT']['PERIODIC_EFIELD']['POLARISATION'] = direction[vector]
            geo_efield['cp2k']['parent_calc_folder'] = self.ctx.ground_state.outputs.remote_folder
            geo_efield['cp2k']['parameters'] = Dict(dict=self.ctx.efield_parameters)
            running_geo_efield = self.submit(Cp2kBaseWorkChain, **geo_efield)
            self.to_context(**{f'efield_{vector}': running_geo_efield})
    
    def results(self):

        def get_dipole_from_output(aiidaout_path):
            my_list = []
            with open(efield0_output, 'r') as my_out:
                for line in my_out.readlines():
                    if 'X=' in line:
                        vector = line.split()
                        my_list.append([vector[1], vector[3],vector[5]])
            return my_list

        ase_structure = self.ctx.ground_state.outputs.output_structure.get_ase()
        volume = ase_structure.get_volume()
        efield0_output = self.ctx.efield0.called[-1].outputs.remote_folder.get_remote_path() +'/aiida.out'
        efield_x_output = self.ctx.efield_x.called[-1].outputs.remote_folder.get_remote_path() +'/aiida.out'
        efield_y_output = self.ctx.efield_y.called[-1].outputs.remote_folder.get_remote_path() +'/aiida.out'
        efield_z_output = self.ctx.efield_z.called[-1].outputs.remote_folder.get_remote_path()+'/aiida.out'
        dipole_e0 = get_dipole_from_output(efield0_output)
        dipole_ex = get_dipole_from_output(efield_x_output)
        dipole_ey = get_dipole_from_output(efield_y_output)
        dipole_ez = get_dipole_from_output(efield_z_output)
        dipole_efield0 = np.linalg.norm(dipole_e0)
        dipole_efieldx = np.linalg.norm(dipole_ex[0])
        dipole_efieldx_relax = np.linalg.norm(dipole_ex[-1])
        dipole_efieldy = np.linalg.norm(dipole_ey[0])
        dipole_efieldy_relax = np.linalg.norm(dipole_ey[-1])
        dipole_efieldz = np.linalg.norm(dipole_ez[0])
        dipole_efieldz_relax = np.linalg.norm(dipole_ez[-1])
        
        eps_inf_xx = (4*np.pi*(np.absolute(dipole_efieldx-dipole_efield0))*0.393456)/(volume*np.power(1.8897259886,3)*0.0005) + 1
        eps_inf_yy = (4*np.pi*np.absolute(dipole_efieldy-dipole_efield0)*0.393456)/(volume*np.power(1.8897259886,3)*0.0005) + 1
        eps_inf_zz = (4*np.pi*np.absolute(dipole_efieldz-dipole_efield0)*0.393456)/(volume*np.power(1.8897259886,3)*0.0005) + 1

        eps_0_xx = (4*np.pi*(np.absolute(dipole_efieldx_relax-dipole_efieldx))*0.393456)/(volume*np.power(1.8897259886,3)*0.0005) + eps_inf_xx
        eps_0_yy = (4*np.pi*(np.absolute(dipole_efieldy_relax-dipole_efieldy))*0.393456)/(volume*np.power(1.8897259886,3)*0.0005) + eps_inf_zz
        eps_0_zz = (4*np.pi*(np.absolute(dipole_efieldz_relax-dipole_efieldz))*0.393456)/(volume*np.power(1.8897259886,3)*0.0005) + eps_inf_yy
        
        results_dict = {
            'eps_inf': [eps_inf_xx,eps_inf_yy,eps_inf_zz],
            'eps_0': [eps_0_xx,eps_0_yy,eps_0_zz],
        }
        aiida_result_dict = Dict(dict=results_dict)
        aiida_result_dict.store()
        self.out('epsilon',aiida_result_dict)
        self.report(f'Cp2kDielectricWorkChain Completed')










        





        




