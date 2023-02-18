import os
import pathlib
import yaml
import copy
import numpy as np

from aiida.engine import WorkChain, ToContext, ExitCode
from aiida.orm import Int, Bool, Code, Dict, List, Str
from aiida.orm import SinglefileData, StructureData, FolderData
from aiida.plugins import CalculationFactory
from aiida_nanotech_empa.workflows.cp2k.cp2k_utils import get_kinds_section, determine_kinds, dict_merge, get_cutoff
from aiida_nanotech_empa.workflows.cp2k.cp2k_utils import get_colvars_section, get_constraints_section

from aiida_nanotech_empa.utils import common_utils

Cp2kCalculation = CalculationFactory("cp2k")


class Cp2kNebWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("code", valid_type=Code)
        spec.input("structure", valid_type=StructureData)
        spec.input("replica_uuids", valid_type=List)
        spec.input("wfn_file_path", valid_type=Str, required=False)
        spec.input("restart_from", valid_type=Str, required=False)
        spec.input("dft_params", valid_type=Dict)
        spec.input("sys_params",valid_type=Dict)
        spec.input("neb_params",valid_type=Dict)        
        spec.input(
            "options",
            valid_type=dict,
            non_db=True,
            help=
            "Define options for the cacluations: walltime, memory, CPUs, etc.")

        #workchain outline
        spec.outline(cls.setup, cls.submit_neb, cls.finalize)
        spec.outputs.dynamic = True

        spec.exit_code(
            390,
            "ERROR_TERMINATION",
            message="One or more steps of the work chain failed.",
        )

    def setup(self):
        self.report("Inspecting input and setting up things")

        self.ctx.files = {
            "basis": SinglefileData(
                file=os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    ".",
                    "data",
                    "BASIS_MOLOPT",
                )
            ),
            "pseudo": SinglefileData(
                file=os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    ".",
                    "data",
                    "POTENTIAL",
                )
            ),
        }        



        self.ctx.sys_params = self.inputs.sys_params.get_dict()
        self.ctx.dft_params = self.inputs.dft_params.get_dict()
        self.ctx.neb_params = self.inputs.neb_params.get_dict()

        self.ctx.n_atoms = len(self.inputs.structure.sites)

        # load input template
        with open(
            pathlib.Path(__file__).parent / "./protocols/neb_protocol.yml",
            encoding="utf-8",
        ) as handle:
            protocols = yaml.safe_load(handle)
            self.ctx.input_dict = copy.deepcopy(protocols[self.ctx.dft_params['protocol']])        

        # vdW section
        if 'vdw' in self.ctx.dft_params:
            if not self.ctx.dft_params['vdw']: self.ctx.input_dict['FORCE_EVAL']['DFT']['XC'].pop('VDW_POTENTIAL')
        else:
            self.ctx.input_dict['FORCE_EVAL']['DFT']['XC'].pop('VDW_POTENTIAL')

        #charge
        if 'charge' in self.ctx.dft_params:
            self.ctx.input_dict['FORCE_EVAL']['DFT']['CHARGE'] = self.ctx.dft_params['charge']
           
        # uks    
        magnetization_per_site = [0 for i in range(self.ctx.n_atoms)]
        if 'uks' in   self.ctx.dft_params:
            if self.ctx.dft_params['uks']:
                magnetization_per_site = self.ctx.dft_params["magnetization_per_site"]
                self.ctx.input_dict['FORCE_EVAL']['DFT']['UKS'] = '.TRUE.'
                self.ctx.input_dict['FORCE_EVAL']['DFT']['MULTIPLICITY'] = self.ctx.dft_params['multiplicity']

        # get initial magnetization
        structure_with_tags, kinds_dict = determine_kinds(
            self.inputs.structure, magnetization_per_site
        )

        ase_atoms = structure_with_tags.get_ase()

        # non periodic systems only NONE and XYZ implemented: TO BE CHECKED FOR NEB!!!!
        if 'periodic' in self.ctx.dft_params:
            if self.ctx.dft_params['periodic'] == 'NONE':
                # make sure cell is big enough for MT poisson solver and center molecule
                if self.ctx.dft_params['protocol'] == "debug":
                    extra_cell = 5.0
                else:
                    extra_cell = 15.0
                ase_atoms.cell = 2 * (np.ptp(ase_atoms.positions, axis=0)) + extra_cell
                ase_atoms.center()

                # Poisson solver
                self.ctx.input_dict['FORCE_EVAL']['SUBSYS']['CELL']['PERIODIC'] = 'NONE'
                self.ctx.input_dict['FORCE_EVAL']['DFT']['POISSON']['PERIODIC'] = 'NONE'
                self.ctx.input_dict['FORCE_EVAL']['DFT']['POISSON']['POISSON_SOLVER'] = 'MT'
            # to be done: more cases 

        # must be after if 'periodic'     
        self.ctx.structure_with_tags = ase_atoms  
        self.ctx.kinds_section = get_kinds_section(kinds_dict, protocol="gpw")   
        dict_merge(self.ctx.input_dict, self.ctx.kinds_section) 

        # replica files with tags must be after structure_with_tags.
        tags = ase_atoms.get_tags()
        self.ctx.files['replica_001_xyz'] = make_geom_file(ase_atoms, 'replica_001.xyz', tags=tags)
        self.ctx.input_dict['MOTION']['BAND']['REPLICA']=[{'COORD_FILE_NAME':'replica_001.xyz'}]
        for uuid in self.inputs.replica_uuids:
            structure = load_node(uuid).get_ase() 
            filename = 'replica_$s.xyz' % str(i +1 ).zfill(3)
            self.ctx.files[filename.replace(".", "_")] = make_geom_file(structure, filename, tags=tags)
            # and update input dictionary.
            self.ctx.input_dict['MOTION']['BAND']['REPLICA'].append({'COORD_FILE_NAME':filename}) 

        # get cutoff.
        cutoff = get_cutoff(structure=self.inputs.structure)

        # overwrite cutoff if given in dft_params.
        if "cutoff" in self.ctx.dft_params:
            cutoff = self.ctx.dft_params["cutoff"]

        self.ctx.input_dict['FORCE_EVAL']['DFT']['MGRID']['CUTOFF'] = cutoff

        # constraints.
        if 'constraints' in self.ctx.sys_params:
            self.ctx.input_dict['MOTION']['CONSTRAINT'] = get_constraints_section(self.ctx.sys_params['constraints'])
        # colvars.
        if 'colvars' in self.ctx.sys_params:
            self.ctx.input_dict['FORCE_EVAL']['SUBSYS'].update(get_colvars_section(self.ctx.sys_params['colvars']))


        # neb parameters.
        for param in ['align_frames','band_type','k_spring','nproc_rep','number_of_replica']
            if param in self.ctx.neb_params:
                    self.ctx.input_dict['MOTION']['BAND'][param.upper()] = self.ctx.neb_params[param]

        if  'nsteps_it' in self.ctx.neb_params:
            self.ctx.input_dict['MOTION']['BAND']['CI_NEB'] = self.ctx.neb_params['nsteps_it']
        if 'optimize_end_points' in self.ctx.neb_params:
            self.ctx.input_dict['MOTION']['BAND']['OPTIMIZE_BAND']['OPTIMIZE_END_POINTS'] = self.ctx.neb_params['optimize_end_points']

        # resources
        self.ctx.options = self.inputs.options
        if self.ctx.dft_params['protocol'] == "debug":
            self.ctx.options = {
                "max_wallclock_seconds": 600,
                "resources": {
                    "num_machines": 3,
                "num_mpiprocs_per_machine": 1,
                "num_cores_per_mpiproc": 1,
                }
                }

        # --------------------------------------------------

    def submit_neb(self):
        self.report("Submitting geometry optimization")

        builder = Cp2kCalculation.get_builder()
        builder.code = self.inputs.code
        builder.structure = StructureData(ase=self.ctx.structure_with_tags)
        builder.file = self.ctx.files

        # resources
        builder.metadata.options = self.ctx.options

        # label
        builder.metadata.label = 'neb'

        # parser
        builder.metadata.options.parser_name = "cp2k_neb_parser"

        # additional retrieved files
        builder.settings = Dict(dict={"additional_retrieve_list": ["*.xyz", "*.out", "*.ener"]})
        # restart wfn
        if "wfn_file_path" in self.inputs:
            builder.parent_calc_folder = self.inputs.wfn_file_path.value

        # cp2k input dictionary
        builder.parameters = Dict(self.ctx.input_dict)

        future = self.submit(builder)
        self.to_context(neb=future)

    def finalize(self):
        self.report("Finalizing.")

        if not self.ctx.neb.is_finished_ok:
            return self.exit_codes.ERROR_TERMINATION

        for i_rep in range(self.ctx.neb_params['number_of_replica']):
            label = "opt_replica_%s" % str(i_rep).zfill(3)
            self.out(label, self.ctx.neb.outputs[label])

        self.out("replica_energies", self.ctx.neb.outputs["replica_energies"])
        self.out("replica_distances", self.ctx.neb.outputs["replica_distances"])

        # Add extras
        struc = self.inputs.structure
        ase_geom = struc.get_ase()
        struc.set_extra('thumbnail',
                            common_utils.thumbnail(ase_struc=ase_geom))
        common_utils.add_extras(struc,'surfaces','neb',self.node.uuid)
        


        return ExitCode(0)
