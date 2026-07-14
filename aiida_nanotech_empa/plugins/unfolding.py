from aiida import common, engine, orm

DEFAULT_WFN_FILENAME = "aiida-RESTART.wfn"
DEFAULT_XYZ_FILENAME = "aiida.coords.xyz"
DEFAULT_CP2K_INPUT_FILENAME = "aiida.inp"
DEFAULT_MATRIX_FILENAME = "aiida-overlap_matrix.out-1_0.Log"
DEFAULT_OUTPUT_FILENAME = "unfolding_bands.npz"
DEFAULT_PDOS_PROJECTION_FILENAME = "unfolding_projections.npz"


class Cp2kUnfoldingCalculation(engine.CalcJob):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input(
            "parent_calc_folder",
            valid_type=orm.RemoteData,
            help="CP2K diagonalization folder containing the WFN, coordinates, and CP2K input.",
        )
        spec.input("primitive_vectors", valid_type=orm.Str)
        spec.input(
            "path",
            valid_type=orm.Str,
            default=lambda: orm.Str("G-K-M-G"),
            required=False,
        )
        spec.input(
            "lattice_type",
            valid_type=orm.Str,
            default=lambda: orm.Str("auto"),
            required=False,
        )
        spec.input("emin", valid_type=orm.Float, required=False)
        spec.input("emax", valid_type=orm.Float, required=False)
        spec.input(
            "wfn_filename",
            valid_type=orm.Str,
            default=lambda: orm.Str(DEFAULT_WFN_FILENAME),
            required=False,
        )
        spec.input(
            "xyz_filename",
            valid_type=orm.Str,
            default=lambda: orm.Str(DEFAULT_XYZ_FILENAME),
            required=False,
        )
        spec.input(
            "cp2k_input_filename",
            valid_type=orm.Str,
            default=lambda: orm.Str(DEFAULT_CP2K_INPUT_FILENAME),
            required=False,
        )
        spec.input(
            "matrix_filename",
            valid_type=orm.Str,
            default=lambda: orm.Str(DEFAULT_MATRIX_FILENAME),
            required=False,
        )
        spec.input(
            "overlap_threshold",
            valid_type=orm.Float,
            default=lambda: orm.Float(1.0e-10),
            required=False,
        )
        spec.input(
            "output_filename",
            valid_type=orm.Str,
            default=lambda: orm.Str(DEFAULT_OUTPUT_FILENAME),
            required=False,
        )
        spec.input(
            "parse_pdos_projections",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            required=False,
        )
        spec.input(
            "pdos_projection_filename",
            valid_type=orm.Str,
            default=lambda: orm.Str(DEFAULT_PDOS_PROJECTION_FILENAME),
            required=False,
        )
        spec.input(
            "pdos_threshold",
            valid_type=orm.Float,
            default=lambda: orm.Float(1.0e-4),
            required=False,
        )
        spec.input(
            "settings",
            valid_type=orm.Dict,
            default=lambda: orm.Dict(dict={}),
            required=False,
        )
        spec.input("metadata.options.withmpi", valid_type=bool, default=False)

    def prepare_for_submission(self, folder):
        settings = self.inputs.settings.get_dict() if "settings" in self.inputs else {}
        output_filename = self.inputs.output_filename.value

        codeinfo = common.CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.cmdline_params = [
            "parent_calc_folder/" + self.inputs.wfn_filename.value,
            "parent_calc_folder/" + self.inputs.matrix_filename.value,
            output_filename,
            "--xyz",
            "parent_calc_folder/" + self.inputs.xyz_filename.value,
            "--cp2k-input",
            "parent_calc_folder/" + self.inputs.cp2k_input_filename.value,
            "--primitive-vectors",
            self.inputs.primitive_vectors.value,
            "--path",
            self.inputs.path.value,
            "--lattice-type",
            self.inputs.lattice_type.value,
            "--overlap-format",
            "log",
            "--overlap-threshold",
            str(self.inputs.overlap_threshold.value),
        ]
        if "emin" in self.inputs:
            codeinfo.cmdline_params.extend(["--emin", str(self.inputs.emin.value)])
        if "emax" in self.inputs:
            codeinfo.cmdline_params.extend(["--emax", str(self.inputs.emax.value)])
        if self.inputs.parse_pdos_projections.value:
            codeinfo.cmdline_params.extend(
                [
                    "--pdos-glob",
                    "parent_calc_folder/aiida-*list*-1.pdos",
                    "--pdos-output",
                    self.inputs.pdos_projection_filename.value,
                    "--pdos-threshold",
                    str(self.inputs.pdos_threshold.value),
                ]
            )

        calcinfo = common.CalcInfo()
        calcinfo.uuid = self.uuid
        calcinfo.codes_info = [codeinfo]
        calcinfo.remote_symlink_list = []
        calcinfo.remote_copy_list = []
        calcinfo.local_copy_list = []
        default_retrieve_list = [output_filename]
        if self.inputs.parse_pdos_projections.value:
            default_retrieve_list.append(self.inputs.pdos_projection_filename.value)
        calcinfo.retrieve_list = settings.pop(
            "additional_retrieve_list", default_retrieve_list
        )

        comp_uuid = self.inputs.parent_calc_folder.computer.uuid
        remote_path = self.inputs.parent_calc_folder.get_remote_path()
        copy_info = (comp_uuid, remote_path, "parent_calc_folder/")
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
