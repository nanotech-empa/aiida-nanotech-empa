from aiida import common, engine, orm


class OverlapCalculation(engine.CalcJob):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("parameters", valid_type=orm.Dict, help="Overlap input parameters")
        spec.input(
            "parent_slab_folder", valid_type=orm.RemoteData, help="slab scf folder"
        )
        spec.input(
            "parent_mol_folder", valid_type=orm.RemoteData, help="molecule scf folder"
        )
        spec.input("settings", valid_type=orm.Dict, help="special settings")

        # Use mpi by default.
        spec.input("metadata.options.withmpi", valid_type=bool, default=True)

    def prepare_for_submission(self, folder):
        """Create the input files from the input nodes passed to this instance of the `CalcJob`.
        :param folder: an `aiida.common.folders.Folder` to temporarily write files on disk
        :return: `aiida.common.datastructures.CalcInfo` instance
        """

        settings = self.inputs.settings.get_dict() if "settings" in self.inputs else {}

        # Create code info.
        codeinfo = common.CodeInfo()

        codeinfo.code_uuid = self.inputs.code.uuid

        param_dict = self.inputs.parameters.get_dict()

        cmdline = []
        for key in param_dict:
            cmdline += [key]
            if param_dict[key] != "":
                if isinstance(param_dict[key], list):
                    cmdline += param_dict[key]
                else:
                    cmdline += [param_dict[key]]

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

        calcinfo.retrieve_list = settings.pop("additional_retrieve_list", [])

        # Symlinks.
        if "parent_slab_folder" in self.inputs:
            comp_uuid = self.inputs.parent_slab_folder.computer.uuid
            remote_path = self.inputs.parent_slab_folder.get_remote_path()
            copy_info = (comp_uuid, remote_path, "parent_slab_folder/")
            if (
                self.inputs.code.computer.uuid == comp_uuid
            ):  # if running on the same computer - make a symlink
                # if not - copy the folder
                calcinfo.remote_symlink_list.append(copy_info)
            else:
                calcinfo.remote_copy_list.append(copy_info)

        if "parent_mol_folder" in self.inputs:
            comp_uuid = self.inputs.parent_mol_folder.computer.uuid
            remote_path = self.inputs.parent_mol_folder.get_remote_path()
            copy_info = (comp_uuid, remote_path, "parent_mol_folder/")
            if (
                self.inputs.code.computer.uuid == comp_uuid
            ):  # if running on the same computer - make a symlink
                # if not - copy the folder
                calcinfo.remote_symlink_list.append(copy_info)
            else:
                calcinfo.remote_copy_list.append(copy_info)

        # Check for left over settings.
        if settings:
            raise common.InputValidationError(
                "The following keys have been found "
                + f"in the settings input node {self.pk}, "
                + "but were not understood: "
                + ",".join(settings.keys())
            )

        return calcinfo
