from aiida import common, engine, orm

DEFAULT_CHARGE_DENSITY_FILENAME = "aiida-ELECTRON_DENSITY-1_0.cube"
BADER_OUTPUT_FILES = ["ACF.dat", "AVF.dat", "BCF.dat"]
# Bader 1.05 defaults, pinned so results do not depend on the installed version.
BADER_OPTIONS = ["-i", "cube", "-b", "neargrid", "-m", "known", "-vac", "off"]


class BaderCalculation(engine.CalcJob):
    """Run Bader charge analysis on a charge-density cube from a CP2K calculation."""

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
            help="Name of the charge-density cube inside 'parent_calc_folder'.",
        )
        spec.input(
            "settings",
            valid_type=orm.Dict,
            default=lambda: orm.Dict(dict={}),
            required=False,
            help="Only 'additional_retrieve_list' is accepted: extra files to "
            "retrieve on top of ACF.dat, AVF.dat and BCF.dat.",
        )
        spec.input("metadata.options.withmpi", valid_type=bool, default=False)
        spec.input(
            "metadata.options.parser_name",
            valid_type=str,
            default="nanotech_empa.bader",
        )

        spec.exit_code(
            300,
            "ERROR_OUTPUT_FILES_MISSING",
            message="One or more Bader output files were not retrieved.",
        )

    def prepare_for_submission(self, folder):
        settings = self.inputs.settings.get_dict()

        codeinfo = common.CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.cmdline_params = BADER_OPTIONS + [
            "parent_calc_folder/" + self.inputs.charge_density_filename.value,
        ]

        calcinfo = common.CalcInfo()
        calcinfo.uuid = self.uuid
        calcinfo.codes_info = [codeinfo]
        calcinfo.remote_symlink_list = []
        calcinfo.remote_copy_list = []
        calcinfo.local_copy_list = []
        calcinfo.retrieve_list = BADER_OUTPUT_FILES + settings.pop(
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
