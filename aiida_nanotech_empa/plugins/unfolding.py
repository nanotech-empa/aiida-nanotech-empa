from aiida import common, engine, orm

from .sparse_overlap import DEFAULT_MATRIX_FILENAME

DEFAULT_WFN_FILENAME = "aiida-RESTART.wfn"
DEFAULT_XYZ_FILENAME = "aiida.coords.xyz"
DEFAULT_CP2K_INPUT_FILENAME = "aiida.inp"
DEFAULT_OUTPUT_FILENAME = "unfolding_bands.npz"
DEFAULT_PATH = "G-K-M-G"
DEFAULT_LATTICE_TYPE = "auto"
LATTICE_TYPES = ("auto", "1d", "square", "rectangular", "hexagonal", "oblique")


def validate_lattice_type(value, _):
    if value.value not in LATTICE_TYPES:
        return f"must be one of {', '.join(LATTICE_TYPES)}."


def validate_primitive_vectors(value, _):
    try:
        vectors = [
            [float(x) for x in row.replace(",", " ").split()]
            for row in value.value.replace(";", "\n").splitlines()
            if row.strip()
        ]
    except ValueError:
        vectors = []
    if not 1 <= len(vectors) <= 2 or any(len(vector) != 3 for vector in vectors):
        return (
            "must be one or two vectors of three numbers, separated by ';' or "
            "newlines; 3D unfolding is not supported."
        )


def validate_energy_window(inputs, emin_key, emax_key):
    if (emin_key in inputs) != (emax_key in inputs):
        return (
            f"'{emin_key}' and '{emax_key}' must be set together: "
            "cp2k-spm-tools ignores a one-sided energy window."
        )
    if emin_key in inputs and inputs[emin_key].value >= inputs[emax_key].value:
        return f"'{emin_key}' must be lower than '{emax_key}'."


class Cp2kUnfoldingCalculation(engine.CalcJob):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input(
            "parent_calc_folder",
            valid_type=orm.RemoteData,
            help="CP2K diagonalization folder containing the WFN, coordinates, and CP2K input.",
        )
        spec.input(
            "primitive_vectors",
            valid_type=orm.Str,
            validator=validate_primitive_vectors,
        )
        spec.input(
            "path",
            valid_type=orm.Str,
            default=lambda: orm.Str(DEFAULT_PATH),
            required=False,
        )
        spec.input(
            "lattice_type",
            valid_type=orm.Str,
            default=lambda: orm.Str(DEFAULT_LATTICE_TYPE),
            required=False,
            validator=validate_lattice_type,
        )
        spec.input(
            "emin",
            valid_type=orm.Float,
            required=False,
            help="Lower bound (eV) of the unfolded energy window, relative to the "
            "middle of the HOMO-LUMO gap. Set together with 'emax'.",
        )
        spec.input(
            "emax",
            valid_type=orm.Float,
            required=False,
            help="Upper bound (eV) of the unfolded energy window, see 'emin'.",
        )
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
            "settings",
            valid_type=orm.Dict,
            default=lambda: orm.Dict(dict={}),
            required=False,
            help="Only 'additional_retrieve_list' is accepted: extra files to "
            "retrieve on top of 'output_filename'.",
        )
        spec.input("metadata.options.withmpi", valid_type=bool, default=False)
        spec.input(
            "metadata.options.parser_name",
            valid_type=str,
            default="nanotech_empa.cp2k_unfolding",
        )

        spec.exit_code(
            300,
            "ERROR_OUTPUT_FILE_MISSING",
            message="The unfolded band output file was not retrieved.",
        )

    def prepare_for_submission(self, folder):
        window_error = validate_energy_window(self.inputs, "emin", "emax")
        if window_error:
            raise common.InputValidationError(window_error)
        settings = self.inputs.settings.get_dict()
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

        calcinfo = common.CalcInfo()
        calcinfo.uuid = self.uuid
        calcinfo.codes_info = [codeinfo]
        calcinfo.remote_symlink_list = []
        calcinfo.remote_copy_list = []
        calcinfo.local_copy_list = []
        calcinfo.retrieve_list = [output_filename] + settings.pop(
            "additional_retrieve_list", []
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
