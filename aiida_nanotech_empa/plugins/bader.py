from aiida import common, engine, orm


DEFAULT_CHARGE_DENSITY_FILENAME = "aiida-ELECTRON_DENSITY-1_0.cube"
DEFAULT_RETRIEVE_LIST = ["ACF.dat", "AVF.dat", "BCF.dat"]


class BaderCalculation(engine.CalcJob):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input(
            "parent_calc_folder",
            valid_type=orm.RemoteData,
            help="CP2K folder containing the charge-density cube.",
        )
        spec.input(
            "charge_density_filename",
            valid_type=orm.Str,
            default=lambda: orm.Str(DEFAULT_CHARGE_DENSITY_FILENAME),
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

        codeinfo = common.CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.cmdline_params = [
            "parent_calc_folder/" + self.inputs.charge_density_filename.value,
        ]

        calcinfo = common.CalcInfo()
        calcinfo.uuid = self.uuid
        calcinfo.codes_info = [codeinfo]
        calcinfo.remote_symlink_list = []
        calcinfo.remote_copy_list = []
        calcinfo.local_copy_list = []
        calcinfo.retrieve_list = settings.pop(
            "additional_retrieve_list", DEFAULT_RETRIEVE_LIST
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
