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
        import os
        import subprocess
        import sys
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


        def _load_inputs():
            with open("banduppy_settings.json", "r") as handle:
                settings = json.load(handle)
            params = settings["parameters"]
            arrays = np.load("banduppy_inputs.npz", allow_pickle=True)
            primitive_kline = (
                arrays["primitive_kline"] if "primitive_kline" in arrays.files else None
            )
            return {
                "params": params,
                "special_labels": settings["special_labels"],
                "mapping": _restore_int_mapping(settings["mapping_data"]),
                "output_npz": settings["output_npz"],
                "supercell_matrix": np.asarray(arrays["supercell_matrix"], dtype=int),
                "kpoints_pbz_full": arrays["kpoints_pbz_full"],
                "kpoints_sbz": arrays["kpoints_sbz"],
                "primitive_kline": primitive_kline,
            }


        def _spin_channels(params):
            spin_channels = params.get("spin_channels")
            if spin_channels is None:
                spin_channel = params.get("spin_channel")
                spin_channels = [spin_channel] if spin_channel is not None else [None]
            return [
                None if channel in (None, "", "none") else str(channel)
                for channel in spin_channels
            ]


        def _channel_key(channel):
            return "none" if channel is None else str(channel)


        def _channel_from_key(channel_key):
            return None if channel_key == "none" else channel_key


        def _band_slice(params):
            ib_start = params.get("ib_start", params.get("IBstart"))
            ib_end = params.get("ib_end", params.get("IBend"))
            ib_start = None if ib_start in (None, "") else int(ib_start)
            ib_end = None if ib_end in (None, "") else int(ib_end)
            return ib_start, ib_end


        def _prefix_path(params):
            prefix = params.get("prefix", "aiida")
            outdir = params.get("outdir", "./out/")
            qe_base = Path("folded_qe_remote")
            prefix_path = Path(outdir)
            if not prefix_path.is_absolute():
                prefix_path = qe_base / prefix_path
            prefix_path = prefix_path / prefix
            save_dir = Path(str(prefix_path) + ".save")
            if not save_dir.exists():
                raise FileNotFoundError(f"Cannot find QE .save directory: {save_dir}")
            return prefix_path


        def _batches(items, batch_size):
            for start in range(0, len(items), batch_size):
                yield items[start:start + batch_size]


        def _local_batch_inputs(full_mapping, batch_indices, full_pbz_kpoints):
            global_pbz_indices = sorted(
                {
                    int(index)
                    for k_index in batch_indices
                    for index_group in full_mapping[int(k_index)].values()
                    for index in index_group
                }
            )
            local_to_global = np.asarray(global_pbz_indices, dtype=int)
            global_to_local = {
                int(global_index): local_index
                for local_index, global_index in enumerate(local_to_global)
            }
            local_mapping = {}
            for k_index in batch_indices:
                local_mapping[int(k_index)] = {
                    local_unique_index: [
                        global_to_local[int(global_index)]
                        for global_index in global_indices
                    ]
                    for local_unique_index, global_indices in enumerate(
                        full_mapping[int(k_index)].values()
                    )
                }
            return full_pbz_kpoints[local_to_global], local_mapping, local_to_global


        def _restore_global_kline(unfolded, local_to_global, primitive_kline):
            if unfolded.size == 0:
                return unfolded
            local_indices = np.rint(unfolded[:, 0]).astype(int)
            valid = (local_indices >= 0) & (local_indices < len(local_to_global))
            unfolded = unfolded.copy()
            unfolded[valid, 0] = local_to_global[local_indices[valid]]
            if primitive_kline is not None:
                global_indices = np.rint(unfolded[:, 0]).astype(int)
                valid = (global_indices >= 0) & (global_indices < len(primitive_kline))
                unfolded[valid, 1] = primitive_kline[global_indices[valid]]
            return unfolded


        def _sort_unfolded_rows(unfolded):
            if unfolded.size == 0:
                return unfolded
            return unfolded[np.lexsort((unfolded[:, 2], unfolded[:, 0]))]


        def _patch_espresso_hdf5_band_slice(ib_start, ib_end):
            if ib_start is None and ib_end is None:
                return False

            import h5py
            from irrep import readfiles as ir_readfiles

            original_parse_header = ir_readfiles.ParserEspresso.parse_header
            original_parse_kpoint = ir_readfiles.ParserEspresso.parse_kpoint

            def _normalized_slice(nbands):
                start = 0 if ib_start is None else int(ib_start)
                stop = nbands if ib_end is None else int(ib_end)
                if start < 0:
                    start = nbands + start
                if stop <= 0:
                    stop = nbands + stop
                if start < 0 or stop > nbands or start >= stop:
                    raise RuntimeError(
                        f"Invalid band slice [{start}, {stop}) for {nbands} bands"
                    )
                return start, stop

            def parse_header(self, spin_channel=None):
                spinpol, ecut0, ef, nkpoints, nbands = original_parse_header(
                    self, spin_channel=spin_channel
                )
                start, stop = _normalized_slice(nbands)
                self._aiida_band_slice = (start, stop, nbands)
                print(
                    "Reading QE HDF5 wavefunctions with band slice "
                    f"[{start}, {stop}) out of {nbands} bands",
                    flush=True,
                )
                return spinpol, ecut0, ef, nkpoints, stop - start

            def parse_kpoint(self, ik, verbosity=0):
                band_slice = getattr(self, "_aiida_band_slice", None)
                if band_slice is None:
                    return original_parse_kpoint(self, ik, verbosity=verbosity)

                kptxml = self.bandstr.findall("ks_energies")[ik]
                if self.spinpol:
                    if self.spin_channel == "up":
                        nb_skip = 0
                        nbands = self.NBin_list[0]
                    else:
                        nb_skip = self.NBin_list[0]
                        nbands = self.NBin_list[1]
                else:
                    nb_skip = 0
                    nbands = self.NBin_list[0]

                start, stop, original_nbands = band_slice
                if nbands != original_nbands:
                    raise RuntimeError(
                        "Band-slice parser saw inconsistent QE band counts: "
                        f"header={original_nbands}, k-point={nbands}"
                    )

                energy_values = np.array(
                    kptxml.find("eigenvalues").text.split(), dtype=float
                )
                energy = energy_values[nb_skip + start:nb_skip + stop]
                energy *= ir_readfiles.Hartree_eV
                npw = int(kptxml.find("npw").text)
                nspinor = 2 if self.spinor else 1

                wfcname = f"wfc{'' if self.spin_channel is None else self.spin_channel}{ik + 1}"
                checked_files = []
                for extension in ["hdf5", "dat"]:
                    for strcase in [str.lower, str.upper]:
                        filename = f"{self.prefix}.save/{strcase(wfcname)}.{extension}"
                        if not os.path.exists(filename):
                            checked_files.append(filename)
                            continue
                        if extension != "hdf5":
                            raise RuntimeError(
                                "Band-sliced QE parsing is implemented for HDF5 "
                                f"wavefunction files only, got {filename}"
                            )
                        with h5py.File(filename, "r") as handle:
                            xk = handle.attrs["xk"]
                            kpt = np.array(xk)
                            miller_indices = handle["MillerIndices"]
                            b_matrix = np.array(
                                [miller_indices.attrs[f"bg{i}"] for i in range(1, 4)]
                            )
                            kg = np.array(miller_indices[::])
                            kpt = kpt.dot(np.linalg.inv(b_matrix))
                            evc = np.asarray(handle["evc"][start:stop, :], dtype=float)
                        wf = evc[:, 0::2] + 1.0j * evc[:, 1::2]
                        wf = wf.reshape((stop - start, npw, nspinor), order="F")
                        return wf, energy, kg, kpt
                raise RuntimeError(f"Wavefunction file not found. Tried files: {checked_files}")

            ir_readfiles.ParserEspresso.parse_header = parse_header
            ir_readfiles.ParserEspresso.parse_kpoint = parse_kpoint
            return True


        def _run_batch_worker(channel_key, batch_indices_text, output_file):
            data = _load_inputs()
            params = data["params"]
            channel = _channel_from_key(channel_key)
            batch_indices = [int(item) for item in batch_indices_text.split(",") if item]
            ib_start, ib_end = _band_slice(params)
            slice_parser_bands = _patch_espresso_hdf5_band_slice(ib_start, ib_end)
            bandstructure_ib_start = None if slice_parser_bands else ib_start
            bandstructure_ib_end = None if slice_parser_bands else ib_end
            local_pbz, local_mapping, local_to_global = _local_batch_inputs(
                data["mapping"], batch_indices, data["kpoints_pbz_full"]
            )
            band_unfold = banduppy.Unfolding(
                supercell=data["supercell_matrix"], print_log=None
            )
            bands = banduppy.BandStructure(
                code="espresso",
                spinor=params.get("spinor"),
                spin_channel=channel,
                prefix=str(_prefix_path(params)),
                kplist=batch_indices,
                IBstart=bandstructure_ib_start,
                IBend=bandstructure_ib_end,
            )
            unfolded, _ = band_unfold.Unfold(
                bands,
                PBZ_kpts_list_full=local_pbz,
                SBZ_kpts_list=data["kpoints_sbz"],
                SBZ_PBZ_kpts_map=local_mapping,
                kline_discontinuity_threshold=float(
                    params.get("kline_discontinuity_threshold", 0.1)
                ),
                save_unfolded_kpts={"save2file": False},
                save_unfolded_bandstr={"save2file": False},
            )
            unfolded = _restore_global_kline(
                unfolded, local_to_global, data["primitive_kline"]
            )
            np.savez(output_file, unfolded=unfolded)


        def _run_batch_subprocess(channel_key, batch_number, batch_indices):
            output_file = f"banduppy_batch_{channel_key}_{batch_number:04d}.npz"
            cmdline = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--batch-worker",
                channel_key,
                ",".join(str(index) for index in batch_indices),
                output_file,
            ]
            subprocess.run(cmdline, check=True)
            with np.load(output_file, allow_pickle=True) as batch_data:
                unfolded = np.asarray(batch_data["unfolded"])
            Path(output_file).unlink(missing_ok=True)
            return unfolded


        def main():
            data = _load_inputs()
            params = data["params"]
            _prefix_path(params)
            spin_channels = _spin_channels(params)
            kpoint_batch_size = max(1, int(params.get("kpoint_batch_size", 1)))
            fermi_energy = params.get("fermi_energy")
            ib_start, ib_end = _band_slice(params)
            sbz_indices = sorted(int(index) for index in data["mapping"])
            unfolded_by_channel = {}
            kline = None

            with open("banduppy_stdout.txt", "w", buffering=1) as stdout, contextlib.redirect_stdout(stdout):
                print(f"BandUPpy k-point batch size: {kpoint_batch_size}")
                print(f"Folded supercell k-points: {len(sbz_indices)}")
                print(f"Band slice: [{ib_start}, {ib_end})")
                for channel in spin_channels:
                    channel_key = _channel_key(channel)
                    channel_parts = []
                    print(f"Spin channel {channel_key}: starting subprocess batches")
                    for batch_number, batch_indices in enumerate(
                        _batches(sbz_indices, kpoint_batch_size), start=1
                    ):
                        print(
                            "  batch "
                            f"{batch_number}: folded k-point indices {batch_indices}",
                            flush=True,
                        )
                        channel_parts.append(
                            _run_batch_subprocess(
                                channel_key, batch_number, batch_indices
                            )
                        )
                    unfolded = (
                        np.concatenate(channel_parts, axis=0)
                        if channel_parts
                        else np.empty((0, 4))
                    )
                    unfolded = _sort_unfolded_rows(unfolded)
                    unfolded_by_channel[channel_key] = unfolded
                    if kline is None:
                        kline = (
                            data["primitive_kline"]
                            if data["primitive_kline"] is not None
                            else np.unique(unfolded[:, 1])
                        )

            first_channel = next(iter(unfolded_by_channel))
            output_arrays = {
                "unfolded_bandstructure": unfolded_by_channel[first_channel],
                "kline": kline,
                "special_labels": json.dumps(data["special_labels"]),
                "supercell_matrix": data["supercell_matrix"],
                "kpoints_pbz_full": data["kpoints_pbz_full"],
                "kpoints_sbz": data["kpoints_sbz"],
                "fermi_energy": np.nan if fermi_energy is None else float(fermi_energy),
                "prefix_path": str(_prefix_path(params)),
                "spin_channels": np.asarray(list(unfolded_by_channel.keys()), dtype=str),
                "kpoint_batch_size": np.asarray(kpoint_batch_size, dtype=int),
                "ib_start": np.asarray(-1 if ib_start is None else ib_start, dtype=int),
                "ib_end": np.asarray(-1 if ib_end is None else ib_end, dtype=int),
            }
            for channel, unfolded in unfolded_by_channel.items():
                output_arrays[f"unfolded_bandstructure_{channel}"] = unfolded
            if data["primitive_kline"] is not None:
                output_arrays["primitive_kline"] = data["primitive_kline"]

            np.savez(data["output_npz"], **output_arrays)


        if __name__ == "__main__":
            if len(sys.argv) > 1 and sys.argv[1] == "--batch-worker":
                _run_batch_worker(sys.argv[2], sys.argv[3], sys.argv[4])
            else:
                main()
        """
    ).lstrip()
