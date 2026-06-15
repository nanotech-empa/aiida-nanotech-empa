import os

import numpy as np
from aiida import engine, orm, plugins

from ...utils import common_utils
from . import cp2k_utils
from .diag_workchain import Cp2kDiagWorkChain
from .geo_opt_workchain import validate_on_unhandled_failure

AfmCalculation = plugins.CalculationFactory("nanotech_empa.afm")


class Cp2kAfmWorkChain(engine.WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("cp2k_code", valid_type=orm.Code)
        spec.input("ppafm_code", valid_type=orm.Code)

        spec.input("structure", valid_type=orm.StructureData)
        spec.input("parent_calc_folder", valid_type=orm.RemoteData, required=False)
        spec.input(
            "protocol",
            valid_type=orm.Str,
            default=lambda: orm.Str("standard"),
            required=False,
            help="Protocol supported by the Cp2kDiagWorkChain.",
        )
        spec.input("dft_params", valid_type=orm.Dict)
        spec.input(
            "options",
            valid_type=dict,
            non_db=True,
            help="Define options for the cacluations: walltime, memory, CPUs, etc.",
        )
        spec.input("ppafm_params", valid_type=orm.Dict)
        spec.input(
            "max_iterations",
            valid_type=orm.Int,
            default=lambda: orm.Int(5),
            required=False,
            help="Maximum number of CP2K restart attempts delegated to cp2k.base.",
        )
        spec.input(
            "clean_workdir",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            required=False,
            help="Clean called CP2K calculation work directories after termination.",
        )
        spec.input(
            "on_unhandled_failure",
            valid_type=orm.Str,
            default=lambda: orm.Str("pause"),
            required=False,
            validator=validate_on_unhandled_failure,
            help="Action for unhandled cp2k.base failures: abort, pause, restart_once, or restart_and_pause.",
        )
        spec.input(
            "pause_on_max_iterations",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(True),
            required=False,
            help="Pause cp2k.base for inspection when restart max_iterations is reached.",
        )

        spec.outline(
            cls.setup,
            cls.run_diag_scf,
            cls.run_afm,
            cls.finalize,
        )

        spec.outputs.dynamic = True

        spec.exit_code(
            390,
            "ERROR_TERMINATION",
            message="One or more steps of the work chain failed.",
        )

    def set_restart_policy(self, builder):
        builder.max_iterations = self.inputs.max_iterations
        builder.clean_workdir = self.inputs.clean_workdir
        builder.on_unhandled_failure = self.inputs.on_unhandled_failure
        builder.pause_on_max_iterations = self.inputs.pause_on_max_iterations

    def setup(self):
        self.report("Setting up workchain")
        structure = self.inputs.structure
        ase_geom = structure.get_ase()
        ase_geom.set_tags(np.zeros(len(ase_geom)))
        n_atoms = len(structure.sites)
        self.ctx.dft_params = self.inputs.dft_params.get_dict()
        if "smear_t" in self.ctx.dft_params:
            added_mos = np.max([100, int(1.2 * n_atoms * 2 / 5.0)])
            self.ctx.dft_params["added_mos"] = added_mos

        self.ctx.files = {
            "geo_no_labels": cp2k_utils.make_geom_file(ase_geom, "geom.xyz"),
            "pp": orm.SinglefileData(
                file=os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    ".",
                    "data",
                    "atomtypes_pp.ini",
                )
            ),
        }

    def run_diag_scf(self):
        self.report("Running CP2K diagonalization SCF")
        builder = Cp2kDiagWorkChain.get_builder()
        builder.cp2k_code = self.inputs.cp2k_code
        builder.structure = self.inputs.structure
        builder.protocol = self.inputs.protocol
        builder.dft_params = orm.Dict(self.ctx.dft_params)
        builder.options = orm.Dict(self.inputs.options)
        self.set_restart_policy(builder)

        # Restart wfn.
        if "parent_calc_folder" in self.inputs:
            builder.parent_calc_folder = self.inputs.parent_calc_folder

        future = self.submit(builder)
        self.to_context(diag_scf=future)

    def run_afm(self):
        self.report("Running ppafm calculation.")
        if not common_utils.check_if_calc_ok(self, self.ctx.diag_scf):
            return self.exit_codes.ERROR_TERMINATION

        afm_inputs = {
            "geo_no_labels": self.ctx.files["geo_no_labels"],
            "metadata": {
                "label": "ppafm",
                "options": {
                    "max_wallclock_seconds": 3600,
                    "resources": {
                        "num_machines": 1,
                        "num_mpiprocs_per_machine": 1,
                        "num_cores_per_mpiproc": 1,
                    },
                },
            },
            "code": self.inputs.ppafm_code,
            "parameters": self.inputs.ppafm_params,
            "parent_calc_folder": self.ctx.diag_scf.outputs.remote_folder,
            "atomtypes": self.ctx.files["pp"],
        }
        self.report("ppafm inputs: " + str(afm_inputs))
        ppafm_future = self.submit(AfmCalculation, **afm_inputs)
        self.to_context(ppafm=ppafm_future)

    def finalize(self):
        retrieved_list = [
            obj.name for obj in self.ctx.ppafm.outputs.retrieved.list_objects()
        ]
        pp_worked = "df.npz" in retrieved_list
        if not pp_worked:
            self.report("AFM calculation did not finish correctly")
            return self.exit_codes.ERROR_TERMINATION

        # Add the workchain pk to the input structure extras.
        common_utils.add_extras(self.inputs.structure, "surfaces", self.node.uuid)
        self.report("Work chain is finished")
