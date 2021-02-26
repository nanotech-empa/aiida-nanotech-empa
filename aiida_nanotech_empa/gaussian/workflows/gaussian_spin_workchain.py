from aiida_nanotech_empa.gaussian.workflows import common

import numpy as np

from aiida.engine import WorkChain, if_, ExitCode
from aiida.orm import Int, Str, Bool, Code, Dict, List
from aiida.orm import StructureData

from aiida.plugins import WorkflowFactory

GaussianBaseWorkChain = WorkflowFactory('gaussian.base')
GaussianCubesWorkChain = WorkflowFactory('gaussian.cubes')

GaussianScfCubesWorkChain = WorkflowFactory('nanotech_empa.gaussian.scf_cubes')
GaussianSpinOptWorkChain = WorkflowFactory('nanotech_empa.gaussian.spin_opt')
GaussianDeltaScfWorkChain = WorkflowFactory('nanotech_empa.gaussian.delta_scf')
GaussianRadicalWorkChain = WorkflowFactory('nanotech_empa.gaussian.radical')


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

            submitted_node = self.submit(
                GaussianSpinOptWorkChain,
                gaussian_code=self.inputs.gaussian_code,
                formchk_code=self.inputs.formchk_code,
                cubegen_code=self.inputs.cubegen_code,
                structure=self.inputs.structure,
                functional=self.inputs.functional,
                basis_set_opt=self.inputs.basis_set_opt,
                basis_set_scf=self.inputs.basis_set_scf,
                multiplicity=Int(mult),
            )
            submitted_node.description = label
            self.to_context(**{label: submitted_node})

    def inspect_opts(self):

        opt_energies = []

        for mult in self.inputs.multiplicity_list:
            label = f"m{mult}_opt"

            # check if everything finished nicely
            if not common.check_if_previous_calc_ok(self, self.ctx[label]):
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
        self.ctx.gs_structure = self.ctx[
            f"m{self.ctx.gs_mult}_opt"].outputs.opt_structure

        self.out("gs_multiplicity", self.ctx.gs_mult)
        self.out("gs_energy", self.ctx.gs_energy)
        self.out("gs_structure", self.ctx.gs_structure)
        self.out("gs_out_params",
                 self.ctx[f"m{self.ctx.gs_mult}_opt"].outputs.scf_out_params)

        self.ctx.gs_scf_calcnode = self.ctx[f"m{self.ctx.gs_mult}_opt"].called[
            0].called[-1]

        return ExitCode(0)

    def submit_next_steps(self):

        cubes_n_occ = 2
        cubes_n_virt = 2
        cubes_isovalues = [0.010, 0.050]

        # ------------------------------------------------------
        self.report("Submitting GS cubes")

        submitted_node = self.submit(
            GaussianCubesWorkChain,
            formchk_code=self.inputs.formchk_code,
            cubegen_code=self.inputs.cubegen_code,
            gaussian_calc_folder=self.ctx.gs_scf_calcnode.outputs.
            remote_folder,
            gaussian_output_params=self.ctx.gs_scf_calcnode.
            outputs['output_parameters'],
            n_occ=Int(cubes_n_occ),
            n_virt=Int(cubes_n_virt),
            cubegen_parser_name='nanotech_empa.gaussian.cubegen_pymol',
            cubegen_parser_params=Dict(dict={'isovalues': cubes_isovalues}),
        )
        submitted_node.description = "gs cubes"
        self.to_context(gs_cubes=submitted_node)

        # ------------------------------------------------------
        self.report("Submitting Delta SCF")

        submitted_node = self.submit(
            GaussianDeltaScfWorkChain,
            gaussian_code=self.inputs.gaussian_code,
            structure=self.ctx.gs_structure,
            functional=self.inputs.functional,
            basis_set=self.inputs.basis_set_scf,
            multiplicity=self.ctx.gs_mult,
            parent_calc_folder=self.ctx.gs_scf_calcnode.outputs.remote_folder,
        )
        submitted_node.description = "delta scf"
        self.to_context(dscf=submitted_node)

        # ------------------------------------------------------
        if self.inputs.functional.value != 'HF':
            self.report("Submitting HF SCF")

            submitted_node = self.submit(
                GaussianScfCubesWorkChain,
                gaussian_code=self.inputs.gaussian_code,
                formchk_code=self.inputs.formchk_code,
                cubegen_code=self.inputs.cubegen_code,
                structure=self.ctx.gs_structure,
                functional=Str("HF"),
                basis_set=self.inputs.basis_set_scf,
                multiplicity=self.ctx.gs_mult,
                do_stable_opt=Bool(True),
                parent_calc_folder=self.ctx.gs_scf_calcnode.outputs.
                remote_folder,
                n_occ=Int(1),
                n_virt=Int(1),
                isosurfaces=List(list=[0.010, 0.050]),
            )
            submitted_node.description = "HF scf"
            self.to_context(gs_hf=submitted_node)

        # ------------------------------------------------------
        self.report("Submitting vertical calculations")

        for mult in self.inputs.multiplicity_list:

            label = f"m{mult}_vert"
            opt_label = f"m{mult}_opt"

            if mult == self.ctx.gs_mult:
                continue

            submitted_node = self.submit(
                GaussianScfCubesWorkChain,
                gaussian_code=self.inputs.gaussian_code,
                formchk_code=self.inputs.formchk_code,
                cubegen_code=self.inputs.cubegen_code,
                structure=self.ctx.gs_structure,
                functional=self.inputs.functional,
                basis_set=self.inputs.basis_set_scf,
                multiplicity=Int(mult),
                parent_calc_folder=self.ctx[opt_label].called[0].called[-1].
                outputs.remote_folder,
                n_occ=Int(cubes_n_occ),
                n_virt=Int(cubes_n_virt),
                isosurfaces=List(list=cubes_isovalues),
            )
            submitted_node.description = label
            self.to_context(**{label: submitted_node})

    def inspect_next_steps(self):

        # ------------------------------------------------------
        if not common.check_if_previous_calc_ok(self, self.ctx.gs_cubes):
            return self.exit_codes.ERROR_TERMINATION

        self.out("gs_cube_images", self.ctx.gs_cubes.outputs.cube_image_folder)

        # ------------------------------------------------------
        if self.inputs.functional.value != 'HF':
            if not common.check_if_previous_calc_ok(self, self.ctx.gs_hf):
                return self.exit_codes.ERROR_TERMINATION

            self.out("gs_hf_out_params", self.ctx.gs_hf.outputs.scf_out_params)
            self.out("gs_hf_cube_images",
                     self.ctx.gs_hf.outputs.cube_image_folder)

        # ------------------------------------------------------
        if not common.check_if_previous_calc_ok(self, self.ctx.dscf):
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
            if not common.check_if_previous_calc_ok(self, self.ctx[label]):
                return self.exit_codes.ERROR_TERMINATION

            vert_energy = self.ctx[label].outputs.scf_energy
            self.out(f"m{mult}_vert_energy", vert_energy)
            self.out(f"m{mult}_vert_out_params",
                     self.ctx[label].outputs.scf_out_params)
            self.out(f"m{mult}_vert_cube_images",
                     self.ctx[label].outputs.cube_image_folder)

        return ExitCode(0)

    def is_gs_oss(self):
        """ Is ground state an open-shell singlet? """
        return self.ctx.gs_mult == 1

    def submit_nat_orb(self):

        self.report("Submitting natural pop analysis")

        submitted_node = self.submit(
            GaussianRadicalWorkChain,
            gaussian_code=self.inputs.gaussian_code,
            parent_calc_folder=self.ctx.gs_scf_calcnode.outputs.remote_folder,
            parent_calc_params=self.ctx.gs_scf_calcnode.outputs.
            output_parameters,
        )
        submitted_node.description = "natural orbitals pop"
        self.to_context(natorb=submitted_node)

        submitted_node = self.submit(
            GaussianRadicalWorkChain,
            gaussian_code=self.inputs.gaussian_code,
            parent_calc_folder=self.ctx.gs_hf.outputs.remote_folder,
            parent_calc_params=self.ctx.gs_hf.outputs.scf_out_params,
        )
        submitted_node.description = "natural orbitals pop HF"
        self.to_context(natorb_hf=submitted_node)

    def inspect_nat_orb(self):

        if not common.check_if_previous_calc_ok(self, self.ctx.natorb):
            return self.exit_codes.ERROR_TERMINATION

        self.out("gs_natorb_params",
                 self.ctx.natorb.outputs.natorb_proc_parameters)

        if not common.check_if_previous_calc_ok(self, self.ctx.natorb_hf):
            return self.exit_codes.ERROR_TERMINATION

        self.out("gs_hf_natorb_params",
                 self.ctx.natorb_hf.outputs.natorb_proc_parameters)

        return ExitCode(0)

    def finalize(self):
        self.report("Finalizing...")
