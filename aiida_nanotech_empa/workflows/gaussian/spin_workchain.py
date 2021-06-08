from aiida_nanotech_empa.utils import common_utils

import numpy as np

from aiida.engine import WorkChain, if_, ExitCode
from aiida.orm import Int, Str, Code, Dict, List, Float
from aiida.orm import StructureData

from aiida.plugins import WorkflowFactory

GaussianBaseWorkChain = WorkflowFactory('gaussian.base')
GaussianCubesWorkChain = WorkflowFactory('gaussian.cubes')

GaussianScfWorkChain = WorkflowFactory('nanotech_empa.gaussian.scf')
GaussianRelaxScfCubesWorkChain = WorkflowFactory(
    'nanotech_empa.gaussian.relax_scf_cubes')
GaussianDeltaScfWorkChain = WorkflowFactory('nanotech_empa.gaussian.delta_scf')
GaussianNatOrbWorkChain = WorkflowFactory('nanotech_empa.gaussian.natorb')


class GaussianSpinWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("gaussian_code", valid_type=Code)
        spec.input("formchk_code", valid_type=Code)
        spec.input("cubegen_code", valid_type=Code)

        spec.input('structure',
                   valid_type=StructureData,
                   required=True,
                   help='input geometry')
        spec.input('functional',
                   valid_type=Str,
                   required=True,
                   help='xc functional')

        spec.input('basis_set_opt',
                   valid_type=Str,
                   required=True,
                   help='basis_set for opt')
        spec.input('basis_set_scf',
                   valid_type=Str,
                   required=True,
                   help='basis_set for scf')

        spec.input('multiplicity_list',
                   valid_type=List,
                   required=True,
                   help='spin multiplicities')

        spec.input(
            'options',
            valid_type=Dict,
            required=False,
            help="Use custom metadata.options instead of the automatic ones.")

        spec.outline(
            cls.submit_opts, cls.inspect_opts, cls.submit_next_steps,
            cls.inspect_next_steps,
            if_(cls.is_gs_oss)(cls.submit_nat_orb,
                               cls.inspect_nat_orb), cls.finalize)

        spec.outputs.dynamic = True

        spec.exit_code(
            390,
            "ERROR_TERMINATION",
            message="One or more steps of the work chain failed.",
        )

    def submit_opts(self):

        # multiplicity 0 means RKS calculation

        for mult in self.inputs.multiplicity_list:
            label = f"m{mult}_opt"

            builder = GaussianRelaxScfCubesWorkChain.get_builder()
            builder.gaussian_code = self.inputs.gaussian_code
            builder.formchk_code = self.inputs.formchk_code
            builder.cubegen_code = self.inputs.cubegen_code
            builder.structure = self.inputs.structure
            builder.functional = self.inputs.functional
            builder.basis_set_opt = self.inputs.basis_set_opt
            builder.basis_set_scf = self.inputs.basis_set_scf
            builder.multiplicity = Int(mult)
            if 'options' in self.inputs:
                builder.options = self.inputs.options

            submitted_node = self.submit(builder)
            submitted_node.description = label
            self.to_context(**{label: submitted_node})

    def inspect_opts(self):

        opt_energies = []

        for mult in self.inputs.multiplicity_list:
            label = f"m{mult}_opt"

            # check if everything finished nicely
            if not common_utils.check_if_calc_ok(self, self.ctx[label]):
                return self.exit_codes.ERROR_TERMINATION

            opt_energy = self.ctx[label].outputs.scf_energy
            opt_energies.append(opt_energy)
            self.out(f"m{mult}_opt_energy", opt_energy)
            self.out(f"m{mult}_opt_structure",
                     self.ctx[label].outputs.opt_structure)
            self.out(f"m{mult}_opt_out_params",
                     self.ctx[label].outputs.scf_out_params)
            self.out(f"m{mult}_opt_cube_images",
                     self.ctx[label].outputs.cube_image_folder)
            self.out(f"m{mult}_opt_cube_planes",
                     self.ctx[label].outputs.cube_planes_array)

        gs_i = np.argmin(opt_energies)

        # if open-shell singlet is degenerate with closed-shell solution, prefer closed-shell
        if self.inputs.multiplicity_list[
                gs_i] == 1 and 0 in self.inputs.multiplicity_list:
            cs_i = self.inputs.multiplicity_list.index(0)
            if np.abs(opt_energies[cs_i].value -
                      opt_energies[gs_i].value) < 1e-6:
                gs_i = cs_i

        self.ctx.gs_mult = Int(self.inputs.multiplicity_list[gs_i]).store()
        self.ctx.gs_energy = opt_energies[gs_i]
        gs_opt_label = f"m{self.ctx.gs_mult.value}_opt"
        self.ctx.gs_structure = self.ctx[gs_opt_label].outputs.opt_structure

        self.out("gs_multiplicity", self.ctx.gs_mult)
        self.out("gs_energy", self.ctx.gs_energy)
        self.out("gs_structure", self.ctx.gs_structure)
        self.out("gs_out_params",
                 self.ctx[gs_opt_label].outputs.scf_out_params)

        self.ctx.gs_scf_calcnode = self.ctx[gs_opt_label].called[0].called[-1]

        return ExitCode(0)

    def submit_next_steps(self):
        # pylint: disable=too-many-statements

        cubes_n_occ = 5
        cubes_n_virt = 5
        cubes_orb_indexes = list(range(-cubes_n_occ + 1, cubes_n_virt + 1))
        cubes_isovalues = [0.010, 0.001]
        cubes_heights = [3.0, 4.0]

        # ------------------------------------------------------
        self.report("Submitting GS cubes")

        builder = GaussianCubesWorkChain.get_builder()
        builder.formchk_code = self.inputs.formchk_code
        builder.cubegen_code = self.inputs.cubegen_code
        builder.gaussian_calc_folder = self.ctx.gs_scf_calcnode.outputs.remote_folder
        builder.gaussian_output_params = self.ctx.gs_scf_calcnode.outputs.output_parameters
        builder.orbital_indexes = List(list=cubes_orb_indexes)
        builder.edge_space = Float(max(cubes_heights))
        builder.cubegen_parser_name = 'nanotech_empa.gaussian.cubegen_pymol'
        builder.cubegen_parser_params = Dict(
            dict={
                'isovalues': cubes_isovalues,
                'heights': cubes_heights,
                'orient_cube': True,
            })

        submitted_node = self.submit(builder)
        submitted_node.description = "gs cubes"
        self.to_context(gs_cubes=submitted_node)

        # ------------------------------------------------------
        self.report("Submitting Delta SCF")

        builder = GaussianDeltaScfWorkChain.get_builder()
        builder.gaussian_code = self.inputs.gaussian_code
        builder.structure = self.ctx.gs_structure
        builder.functional = self.inputs.functional
        builder.basis_set = self.inputs.basis_set_scf
        builder.multiplicity = self.ctx.gs_mult
        builder.parent_calc_folder = self.ctx.gs_scf_calcnode.outputs.remote_folder
        if 'options' in self.inputs:
            builder.options = self.inputs.options

        submitted_node = self.submit(builder)
        submitted_node.description = "delta scf"
        self.to_context(dscf=submitted_node)

        # ------------------------------------------------------
        self.report("Submitting vertical calculations")

        for mult in self.inputs.multiplicity_list:

            label = f"m{mult}_vert"
            opt_label = f"m{mult}_opt"

            if mult == self.ctx.gs_mult:
                continue

            builder = GaussianScfWorkChain.get_builder()
            builder.gaussian_code = self.inputs.gaussian_code
            builder.formchk_code = self.inputs.formchk_code
            builder.cubegen_code = self.inputs.cubegen_code
            builder.structure = self.ctx.gs_structure
            builder.functional = self.inputs.functional
            builder.basis_set = self.inputs.basis_set_scf
            builder.multiplicity = Int(mult)
            builder.parent_calc_folder = self.ctx[
                opt_label].outputs.remote_folder
            builder.n_occ = Int(cubes_n_occ)
            builder.n_virt = Int(cubes_n_virt)
            builder.cubegen_parser_params = Dict(
                dict={
                    'isovalues': cubes_isovalues,
                    'heights': cubes_heights,
                    'orient_cube': True,
                })

            if 'options' in self.inputs:
                builder.options = self.inputs.options

            submitted_node = self.submit(builder)
            submitted_node.description = label
            self.to_context(**{label: submitted_node})

    def inspect_next_steps(self):

        # ------------------------------------------------------
        if not common_utils.check_if_calc_ok(self, self.ctx.gs_cubes):
            return self.exit_codes.ERROR_TERMINATION

        self.out("gs_cube_images", self.ctx.gs_cubes.outputs.cube_image_folder)
        self.out("gs_cube_planes", self.ctx.gs_cubes.outputs.cube_planes_array)

        # ------------------------------------------------------
        if not common_utils.check_if_calc_ok(self, self.ctx.dscf):
            return self.exit_codes.ERROR_TERMINATION

        self.out("gs_ionization_potential",
                 self.ctx.dscf.outputs.ionization_potential)
        self.out("gs_electron_affinity",
                 self.ctx.dscf.outputs.electron_affinity)

        # ------------------------------------------------------
        for mult in self.inputs.multiplicity_list:

            label = f"m{mult}_vert"

            if mult == self.ctx.gs_mult:
                continue

            # check if everything finished nicely
            if not common_utils.check_if_calc_ok(self, self.ctx[label]):
                return self.exit_codes.ERROR_TERMINATION

            vert_energy = self.ctx[label].outputs.energy_ev
            self.out(f"m{mult}_vert_energy", vert_energy)
            self.out(f"m{mult}_vert_out_params",
                     self.ctx[label].outputs.output_parameters)
            self.out(f"m{mult}_vert_cube_images",
                     self.ctx[label].outputs.cube_image_folder)
            self.out(f"m{mult}_vert_cube_planes",
                     self.ctx[label].outputs.cube_planes_array)

        return ExitCode(0)

    def is_gs_oss(self):
        """ Is ground state an open-shell singlet? """
        return self.ctx.gs_mult == 1

    def submit_nat_orb(self):

        self.report("Submitting natural pop analysis")

        builder = GaussianNatOrbWorkChain.get_builder()
        builder.gaussian_code = self.inputs.gaussian_code
        builder.parent_calc_folder = self.ctx.gs_scf_calcnode.outputs.remote_folder
        builder.parent_calc_params = self.ctx.gs_scf_calcnode.outputs.output_parameters
        if 'options' in self.inputs:
            builder.options = self.inputs.options

        submitted_node = self.submit(builder)
        submitted_node.description = "natural orbitals pop"
        self.to_context(natorb=submitted_node)

    def inspect_nat_orb(self):

        if not common_utils.check_if_calc_ok(self, self.ctx.natorb):
            return self.exit_codes.ERROR_TERMINATION

        self.out("gs_natorb_params",
                 self.ctx.natorb.outputs.natorb_proc_parameters)

        return ExitCode(0)

    def finalize(self):
        self.report("Finalizing...")
