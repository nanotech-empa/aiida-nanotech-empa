import copy
import zipfile
import zlib

import numpy as np
from aiida import engine, orm, plugins

from ...utils import common_utils
from . import cp2k_utils

Cp2kDiagWorkChain = plugins.WorkflowFactory("nanotech_empa.cp2k.diag")
StmCalculation = plugins.CalculationFactory("nanotech_empa.stm")


class Cp2kStmWorkChain(engine.WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("cp2k_code", valid_type=orm.Code)
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
        spec.input("spm_code", valid_type=orm.Code)
        spec.input("spm_params", valid_type=orm.Dict)
        spec.input(
            "options",
            valid_type=dict,
            non_db=True,
            help="Define options for the cacluations: walltime, memory, CPUs, etc.",
        )
        cp2k_utils.add_restart_policy_inputs(spec)

        spec.outline(
            cls.setup,
            cls.run_diag_scf,
            cls.run_stm,
            cls.finalize,
        )

        spec.outputs.dynamic = True

        spec.exit_code(
            390,
            "ERROR_TERMINATION",
            message="One or more steps of the work chain failed.",
        )

    def setup(self):
        self.report("Setting up workchain")
        structure = self.inputs.structure
        self.ctx.n_atoms = len(structure.sites)
        emax = float(self.inputs.spm_params.get_dict()["--energy_range"][1])
        added_mos = np.max([100, int(1.2 * self.ctx.n_atoms * emax / 5.0)])
        self.ctx.dft_params = self.inputs.dft_params.get_dict()
        self.ctx.dft_params["added_mos"] = added_mos

    def run_diag_scf(self):
        self.report("Running CP2K diagonalization SCF")
        builder = Cp2kDiagWorkChain.get_builder()
        builder.cp2k_code = self.inputs.cp2k_code
        builder.structure = self.inputs.structure
        builder.protocol = self.inputs.protocol
        builder.dft_params = orm.Dict(self.ctx.dft_params)
        builder.options = orm.Dict(self.inputs.options)
        cp2k_utils.set_restart_policy(self.inputs, builder)

        # Restart wfn, if requested.
        if "parent_calc_folder" in self.inputs:
            builder.parent_calc_folder = self.inputs.parent_calc_folder

        future = self.submit(builder)
        self.to_context(diag_scf=future)

    def run_stm(self):
        self.report("STM calculation")
        if not common_utils.check_if_calc_ok(self, self.ctx.diag_scf):
            return self.exit_codes.ERROR_TERMINATION
        inputs = {}
        inputs["metadata"] = {}
        inputs["metadata"]["label"] = "stm"
        inputs["code"] = self.inputs.spm_code
        spm_params = cp2k_utils.update_legacy_basis_parameter(
            self.inputs.spm_params.get_dict(),
            self.ctx.dft_params,
            self.inputs.structure.get_symbols_set(),
        )
        inputs["parameters"] = orm.Dict(spm_params)
        inputs["parent_calc_folder"] = self.ctx.diag_scf.outputs.remote_folder

        inputs["metadata"]["options"] = copy.deepcopy(self.inputs.options)

        # Need to make an explicit instance for the node to be stored to AiiDA.
        settings = orm.Dict({"additional_retrieve_list": ["stm.npz"]})
        inputs["settings"] = settings

        future = self.submit(StmCalculation, **inputs)
        return engine.ToContext(stm=future)

    def finalize(self):
        if not self.ctx.stm.is_finished_ok:
            self.report(
                "STM helper did not finish successfully "
                f"(exit status: {self.ctx.stm.exit_status})."
            )
            return self.exit_codes.ERROR_TERMINATION
        if "retrieved" not in self.ctx.stm.outputs:
            self.report("STM helper produced no retrieved results.")
            return self.exit_codes.ERROR_TERMINATION
        repository = self.ctx.stm.outputs.retrieved.base.repository
        if "stm.npz" not in repository.list_object_names():
            self.report("STM helper produced no stm.npz result.")
            return self.exit_codes.ERROR_TERMINATION
        try:
            with repository.open("stm.npz", mode="rb") as handle:
                # Check every ZIP member without unpickling legacy metadata.
                with zipfile.ZipFile(handle) as archive:
                    if archive.testzip() is not None:
                        raise ValueError("archive checksum failed")
                handle.seek(0)
                with np.load(handle, allow_pickle=False) as result:
                    required = {
                        "stm_general_info",
                        "stm_series_info",
                        "stm_series_data",
                    }
                    if not required.issubset(result.files):
                        raise ValueError("required STM result entries are missing")
                    maps = result["stm_series_data"]
                    if maps.ndim != 4 or not maps.size or maps.dtype.kind not in "fiu":
                        raise ValueError(
                            "STM map array is empty or has an invalid format"
                        )
        except (OSError, ValueError, EOFError, zipfile.BadZipFile, zlib.error) as exc:
            self.report(
                f"STM helper produced an unreadable or empty stm.npz result: {exc}"
            )
            return self.exit_codes.ERROR_TERMINATION

        self.out("dft_output_parameters", self.ctx.diag_scf.outputs.output_parameters)

        # Add the workchain uuid to the input structure extras.
        common_utils.add_extras(self.inputs.structure, "surfaces", self.node.uuid)
        self.report("Work chain is finished")
