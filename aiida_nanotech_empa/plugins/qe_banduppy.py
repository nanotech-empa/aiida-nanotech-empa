import json
import textwrap

import numpy as np
from aiida import common, engine, orm

DEFAULT_INPUT_NPZ = "banduppy_inputs.npz"
DEFAULT_SETTINGS_JSON = "banduppy_settings.json"
DEFAULT_RUN_SCRIPT = "run_banduppy_qe.py"
DEFAULT_OUTPUT_NPZ = "unfolding_bands.npz"


class QeBanduppyCalculation(engine.CalcJob):
    """Run BandUPpy on a Quantum ESPRESSO bands calculation remote folder."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input(
            "folded_qe_remote_folder",
            valid_type=orm.RemoteData,
            help="Remote folder of the QE bands calculation on folded supercell k-points.",
        )
        spec.input("parameters", valid_type=orm.Dict)
        spec.input("mapping_arrays", valid_type=orm.ArrayData)
        spec.input("mapping_data", valid_type=orm.Dict)
        spec.input("special_labels", valid_type=orm.Dict)
        spec.input(
            "settings",
            valid_type=orm.Dict,
            default=lambda: orm.Dict(dict={}),
            required=False,
        )
        spec.input("metadata.options.withmpi", valid_type=bool, default=False)

    def prepare_for_submission(self, folder):
        settings = self.inputs.settings.get_dict() if "settings" in self.inputs else {}
        parameters = self.inputs.parameters.get_dict()

        with folder.open(DEFAULT_SETTINGS_JSON, "w") as handle:
            json.dump(
                {
                    "parameters": parameters,
                    "mapping_data": self.inputs.mapping_data.get_dict(),
                    "special_labels": self.inputs.special_labels.get_dict(),
                    "output_npz": parameters.get("output_npz", DEFAULT_OUTPUT_NPZ),
                },
                handle,
            )

        arrays = self.inputs.mapping_arrays
        npz_arrays = {
            "kpoints_pbz_full": arrays.get_array("kpoints_pbz_full"),
            "kpoints_sbz": arrays.get_array("kpoints_sbz"),
            "supercell_matrix": arrays.get_array("supercell_matrix"),
        }
        if "primitive_kline" in arrays.get_arraynames():
            npz_arrays["primitive_kline"] = arrays.get_array("primitive_kline")
        with folder.open(DEFAULT_INPUT_NPZ, "wb") as handle:
            np.savez(handle, **npz_arrays)

        with folder.open(DEFAULT_RUN_SCRIPT, "w") as handle:
            handle.write(_runner_script())

        codeinfo = common.CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.cmdline_params = [DEFAULT_RUN_SCRIPT]

        calcinfo = common.CalcInfo()
        calcinfo.uuid = self.uuid
        calcinfo.codes_info = [codeinfo]
        calcinfo.remote_symlink_list = []
        calcinfo.remote_copy_list = []
        calcinfo.local_copy_list = []

        output_npz = parameters.get("output_npz", DEFAULT_OUTPUT_NPZ)
        calcinfo.retrieve_list = settings.pop(
            "additional_retrieve_list",
            [
                output_npz,
                "banduppy_stdout.txt",
            ],
        )

        comp_uuid = self.inputs.folded_qe_remote_folder.computer.uuid
        remote_path = self.inputs.folded_qe_remote_folder.get_remote_path()
        copy_info = (comp_uuid, remote_path, "folded_qe_remote/")
        if self.inputs.code.computer.uuid == comp_uuid:
            calcinfo.remote_symlink_list.append(copy_info)
        else:
            calcinfo.remote_copy_list.append(copy_info)

        if settings:
            raise common.InputValidationError(
                "The following keys have been found in settings but were not understood: "
                + ",".join(settings.keys())
            )

        return calcinfo


def _runner_script():
    return textwrap.dedent(
        r"""
        import contextlib
        import json
        from pathlib import Path

        import numpy as np

        import banduppy


        def _restore_int_mapping(mapping):
            return {
                int(k_index): {
                    int(unique_index): [int(item) for item in k_indices]
                    for unique_index, k_indices in unique_map.items()
                }
                for k_index, unique_map in mapping.items()
            }


        with open("banduppy_settings.json", "r") as handle:
            settings = json.load(handle)
        params = settings["parameters"]
        special_labels = settings["special_labels"]
        sbz_pbz_mapping = _restore_int_mapping(settings["mapping_data"])
        output_npz = settings["output_npz"]

        arrays = np.load("banduppy_inputs.npz", allow_pickle=True)
        supercell_matrix = np.asarray(arrays["supercell_matrix"], dtype=int)
        kpoints_pbz_full = arrays["kpoints_pbz_full"]
        kpoints_sbz = arrays["kpoints_sbz"]
        primitive_kline = arrays["primitive_kline"] if "primitive_kline" in arrays.files else None

        prefix = params.get("prefix", "aiida")
        outdir = params.get("outdir", "./out/")
        spinor = params.get("spinor")
        spin_channels = params.get("spin_channels")
        if spin_channels is None:
            spin_channel = params.get("spin_channel")
            spin_channels = [spin_channel] if spin_channel is not None else [None]
        spin_channels = [None if channel in (None, "", "none") else str(channel) for channel in spin_channels]
        discontinuity = float(params.get("kline_discontinuity_threshold", 0.1))
        fermi_energy = params.get("fermi_energy")

        qe_base = Path("folded_qe_remote")
        prefix_path = Path(outdir)
        if not prefix_path.is_absolute():
            prefix_path = qe_base / prefix_path
        prefix_path = prefix_path / prefix
        save_dir = Path(str(prefix_path) + ".save")
        if not save_dir.exists():
            raise FileNotFoundError(f"Cannot find QE .save directory: {save_dir}")

        unfolded_by_channel = {}
        kline = None
        with open("banduppy_stdout.txt", "w") as stdout, contextlib.redirect_stdout(stdout):
            for channel in spin_channels:
                suffix = "" if channel is None else f"_{channel}"
                band_unfold = banduppy.Unfolding(supercell=supercell_matrix, print_log=None)
                bands = banduppy.BandStructure(
                    code="espresso",
                    spinor=spinor,
                    spin_channel=channel,
                    prefix=str(prefix_path),
                )
                unfolded, channel_kline = band_unfold.Unfold(
                    bands,
                    PBZ_kpts_list_full=kpoints_pbz_full,
                    SBZ_kpts_list=kpoints_sbz,
                    SBZ_PBZ_kpts_map=sbz_pbz_mapping,
                    kline_discontinuity_threshold=discontinuity,
                    save_unfolded_kpts={
                        "save2file": True,
                        "fdir": ".",
                        "fname": "kpoints_unfolded",
                        "fname_suffix": suffix,
                    },
                    save_unfolded_bandstr={
                        "save2file": True,
                        "fdir": ".",
                        "fname": "bandstructure_unfolded",
                        "fname_suffix": suffix,
                    },
                )
                if primitive_kline is not None:
                    indices = np.rint(unfolded[:, 0]).astype(int)
                    valid = (indices >= 0) & (indices < len(primitive_kline))
                    unfolded = unfolded.copy()
                    unfolded[valid, 1] = primitive_kline[indices[valid]]
                unfolded_by_channel["none" if channel is None else channel] = unfolded
                if kline is None:
                    kline = channel_kline

        first_channel = next(iter(unfolded_by_channel))
        output_arrays = {
            "unfolded_bandstructure": unfolded_by_channel[first_channel],
            "kline": kline,
            "special_labels": json.dumps(special_labels),
            "supercell_matrix": supercell_matrix,
            "kpoints_pbz_full": kpoints_pbz_full,
            "kpoints_sbz": kpoints_sbz,
            "fermi_energy": np.nan if fermi_energy is None else float(fermi_energy),
            "prefix_path": str(prefix_path),
            "spin_channels": np.asarray(list(unfolded_by_channel.keys()), dtype=str),
        }
        for channel, unfolded in unfolded_by_channel.items():
            output_arrays[f"unfolded_bandstructure_{channel}"] = unfolded
        if primitive_kline is not None:
            output_arrays["primitive_kline"] = primitive_kline

        np.savez(output_npz, **output_arrays)
        """
    ).lstrip()
