from aiida import engine, orm


class CubeHandlerCalculation(engine.CalcJob):

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input(
            "parameters", valid_type=orm.Dict, help="CubeHandler input parameters."
        )
        spec.input(
            "parent_calc_folder",
            valid_type=orm.RemoteData,
            help="Parent folder containing original cube files.",
        )

    def prepare_for_submission(self, folder):
        """Create the input files from the input nodes passed to this instance of the `CalcJob`."""

        # Create code info.
        codeinfo = orm.CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        self.inputs.parameters.get_dict()
        cmdline = []
        codeinfo.cmdline_params = cmdline

        # Create calc info.
        calcinfo = orm.CalcInfo()
        calcinfo.uuid = self.uuid
        calcinfo.cmdline_params = codeinfo.cmdline_params
        calcinfo.codes_info = [codeinfo]

        # File lists.
        calcinfo.remote_symlink_list = []
        calcinfo.local_copy_list = []
        calcinfo.remote_copy_list = []
        calcinfo.retrieve_list = ["*.cube"]

        # Symlinks.
        if "parent_calc_folder" in self.inputs:
            comp_uuid = self.inputs.parent_calc_folder.computer.uuid
            remote_path = self.inputs.parent_calc_folder.get_remote_path()
            copy_info = (comp_uuid, remote_path, "parent_calc_folder/")
            if self.inputs.code.computer.uuid == comp_uuid:
                calcinfo.remote_symlink_list.append(copy_info)
            else:
                calcinfo.remote_copy_list.append(copy_info)

        return calcinfo
