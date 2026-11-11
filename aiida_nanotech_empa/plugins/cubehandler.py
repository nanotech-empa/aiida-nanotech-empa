import json
from pathlib import PurePosixPath

import yaml
from aiida import common, engine, orm


def json_companion_files(parameters):
    """Serialize explicit JSON companions inside a new job's input directory."""
    values = parameters.get("json_files", {})
    if not isinstance(values, dict):
        raise common.InputValidationError(
            "json_files must be a filename-to-JSON mapping"
        )
    result = {}
    for name, value in values.items():
        if (
            not isinstance(name, str)
            or PurePosixPath(name).name != name
            or not name.endswith(".json")
            or name.startswith(".")
        ):
            raise common.InputValidationError(
                "JSON companion filenames must be plain .json basenames"
            )
        try:
            result[name] = json.dumps(value, allow_nan=False, sort_keys=True)
        except (ValueError, TypeError) as exc:
            raise common.InputValidationError(
                "JSON companions must contain finite JSON values"
            ) from exc
    return result


class CubeHandlerCalculation(engine.CalcJob):
    _DEFAULT_INPUT_FILE = "aiida.inp"

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input(
            "parameters", valid_type=orm.Dict, help="CubeHandler input parameters."
        )
        spec.input_namespace(
            "parent_folders",
            valid_type=orm.RemoteData,
            required=False,
            help="Parent folders containing original cube files.",
        )
        spec.input("metadata.options.withmpi", valid_type=bool, default=False)

    def prepare_for_submission(self, folder):
        """Create the input files from the input nodes passed to this instance of the `CalcJob`."""

        parameters = self.inputs.parameters.get_dict()
        companions = json_companion_files(parameters)
        for name, content in companions.items():
            with folder.open(name, "w") as handle:
                handle.write(content)

        # Create input file.
        with folder.open(self._DEFAULT_INPUT_FILE, "w") as infile:
            yaml.dump(parameters, infile)

        # Create code info.
        codeinfo = common.CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        self.inputs.parameters.get_dict()
        cmdline = ["run", "aiida.inp"]
        codeinfo.cmdline_params = cmdline

        # Create calc info.
        calcinfo = common.CalcInfo()
        calcinfo.uuid = self.uuid
        calcinfo.cmdline_params = codeinfo.cmdline_params
        calcinfo.codes_info = [codeinfo]

        # File lists.
        calcinfo.remote_symlink_list = []
        calcinfo.local_copy_list = []
        calcinfo.remote_copy_list = []
        calcinfo.retrieve_list = ["out_cubes"]

        # Symlinks.
        for folder_name, folder_node in self.inputs.parent_folders.items():
            comp_uuid = folder_node.computer.uuid
            remote_path = folder_node.get_remote_path()
            copy_info = (comp_uuid, remote_path, folder_name)
            if self.inputs.code.computer.uuid == comp_uuid:
                calcinfo.remote_symlink_list.append(copy_info)
            else:
                calcinfo.remote_copy_list.append(copy_info)

        return calcinfo
