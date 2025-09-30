import yaml
from aiida import common, engine, orm


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

        # Create input file.
        with folder.open(self._DEFAULT_INPUT_FILE, "w") as infile:
            yaml.dump(self.inputs.parameters.get_dict(), infile)

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
        calcinfo.retrieve_list = ["*.cube"]

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
