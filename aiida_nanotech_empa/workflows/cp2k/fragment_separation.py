import pathlib

from aiida import engine, orm, plugins

from ...utils import analyze_structure, common_utils, split_structure
from . import cp2k_utils

StructureData = plugins.DataFactory("core.structure")
Cp2kBaseWorkChain = plugins.WorkflowFactory("cp2k.base")

DATA_DIR = pathlib.Path(__file__).parent.absolute() / "data"


class Cp2kFragmentSeparationWorkChain(engine.WorkChain):
    """WorkChain to compute adsoprtion energy for a molecule on a substrate."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("code", valid_type=orm.Code)

        # Specify the input structure and its fragments.
        spec.input(
            "structure", valid_type=StructureData, help="A molecule on a substrate."
        )

        spec.input_namespace(
            "fragments",
            valid_type=orm.List,
            help="List of indices of atoms defining individual fragments.",
        )

        spec.input(
            "protocol",
            valid_type=orm.Str,
            default=lambda: orm.Str("standard"),
            required=False,
            help="Protocol used by the work chain (geo_opt_protocol).",
        )

        spec.input(
            "dft_params",
            valid_type=orm.Dict,
            help="""
        multiplicities, dictionary: multiplicity of each fragment. Use 'all' to specify the multiplicity of the whole system.
        magnetization_per_site: Magnetization per site of the whole system.
            Magnetization per site of the fragments will be extracted automatically.
        charges, dictionary: Charges of each fragment. No need to specify the charge of the full system
            as it would be computed automatically.
        uks: Use unrestricted Kohn-Sham.
        """,
        )

        # Fixed atoms and magnetization per site defined for the whole system. Information for the fragments will extracted automatically.
        spec.input(
            "fixed_atoms",
            valid_type=orm.List,
            required=False,
            help="Fixed atoms of the whole system. Fixed atoms of the fragments will be extracted automatically.",
        )

        spec.input_namespace(
            "auxilary_dictionaries",
            valid_type=orm.Dict,
            help="Dictionaries to append to the CP2K input dictionary right before the job submission. "
            "It is useful to add constraints and collective variables to the input.",
        )

        spec.input_namespace(
            "options",
            valid_type=int,
            non_db=True,
            required=False,
            help="Define options for the cacluations: walltime, memory, CPUs, etc.",
        )

        # in case wfn for the whole system is available and matches uks/rks parameters
        spec.input("parent_calc_folder", valid_type=orm.RemoteData, required=False)

        # Outline.
        spec.outline(cls.setup, cls.run_scfs, cls.run_geo_opts, cls.finalize)

        # Dynamic outputs.
        spec.outputs.dynamic = True

        # Exit code.
        spec.exit_code(
            300,
            "ERROR_TERMINATION",
            message="One or more steps of the work chain failed.",
        )

    def setup(self):
        """Setup the work chain."""

        self.report("Inspecting input and setting things up.")

        n_atoms = len(self.inputs.structure.sites)
        if "magnetization_per_site" in self.inputs.dft_params:
            n_mags = len(list(self.inputs.dft_params["magnetization_per_site"]))
            if n_mags not in (0, n_atoms):
                self.report(
                    "If set, magnetization_per_site needs a value for every atom."
                )
                return self.exit_codes.ERROR_TERMINATION

        self.ctx.cutoff = cp2k_utils.get_cutoff(structure=self.inputs.structure)

        self.ctx.file = {
            "basis": orm.SinglefileData(file=DATA_DIR / "BASIS_MOLOPT"),
            "pseudo": orm.SinglefileData(file=DATA_DIR / "POTENTIAL"),
            "dftd3": orm.SinglefileData(
                file=DATA_DIR / "dftd3.dat"
            ),  # TODO: make vdw optional
        }

        return engine.ExitCode(0)

    def run_scfs(self):
        """Run SCF calculation for the whole system and for fragments."""
        # Firt run SCF for the whole system and its fragments.
        for inputs in split_structure.split_structure(
            structure=self.inputs.structure,
            fixed_atoms=self.inputs.fixed_atoms,
            magnetization_per_site=self.inputs.dft_params["magnetization_per_site"]
            if "magnetization_per_site" in self.inputs.dft_params
            else None,
            fragments=self.inputs.fragments,
        ):
            # Re-loading the input dictionary for the given protocol.
            input_dict = cp2k_utils.load_protocol(
                fname="geo_opt_protocol.yml",
                protocol=self.inputs.protocol.value,
            )

            self.report(
                f"""Running SCF for the fragment '{inputs['label']}' consisting of {len(inputs['structure'].sites)} atoms, """
                f"""where {analyze_structure.list_to_string_range(inputs['fixed_atoms']) or 'None'} atoms are fixed."""
            )

            # Fragment's label.
            fragment = inputs["label"]

            # Generic inputs that are always the same.
            builder = Cp2kBaseWorkChain.get_builder()
            builder.cp2k.code = self.inputs.code
            builder.cp2k.metadata.options.parser_name = "cp2k_advanced_parser"

            # restart wfn in case of fragment 'all'
            if inputs["label"] == "all":
                if "parent_calc_folder" in self.inputs:
                    builder.cp2k.parent_calc_folder = self.inputs.parent_calc_folder

            # If options are defined for a specific fragment, use them.
            if "options" in self.inputs and fragment in self.inputs.options:
                builder.cp2k.metadata.options = self.inputs.options[fragment]

            builder.cp2k.file = self.ctx.file
            input_dict["GLOBAL"]["RUN_TYPE"] = "ENERGY"
            input_dict["FORCE_EVAL"]["DFT"]["MGRID"]["CUTOFF"] = self.ctx.cutoff

            # Always compute charge density with STRIDE 2 2 2 for the SCF part of the work chain.
            input_dict["FORCE_EVAL"]["DFT"]["PRINT"]["E_DENSITY_CUBE"][
                "STRIDE"
            ] = "2 2 2"

            # If charge is set, add it to the corresponding section of the input.
            if (
                "charges" in self.inputs.dft_params
                and fragment in self.inputs.dft_params["charges"]
            ):
                input_dict["FORCE_EVAL"]["DFT"]["CHARGE"] = self.inputs.dft_params[
                    "charges"
                ][fragment]

            if "uks" in self.inputs.dft_params and self.inputs.dft_params["uks"]:
                input_dict["FORCE_EVAL"]["DFT"]["UKS"] = ".TRUE."

                # If the multiplicity is set, add it to the corresponding section of the input.
                if (
                    "multiplicities" in self.inputs.dft_params
                    and fragment in self.inputs.dft_params["multiplicities"]
                ):
                    input_dict["FORCE_EVAL"]["DFT"][
                        "MULTIPLICITY"
                    ] = self.inputs.dft_params["multiplicities"][fragment]

                # Adding magnetisation tags to the structure and to the CP2K input dictionary.
                structure, kinds_dict = cp2k_utils.determine_kinds(
                    inputs["structure"], inputs["magnetization_per_site"]
                )
            else:
                structure, kinds_dict = cp2k_utils.determine_kinds(inputs["structure"])

            kinds_section = cp2k_utils.get_kinds_section(kinds_dict, protocol="gpw")
            cp2k_utils.dict_merge(input_dict, kinds_section)
            builder.cp2k.structure = structure

            # Fixed atoms
            if "fixed_atoms" in self.inputs:
                input_dict["MOTION"]["CONSTRAINT"]["FIXED_ATOMS"][
                    "LIST"
                ] = analyze_structure.list_to_string_range(inputs["fixed_atoms"])

            # Finally, append auxilary dictionaries to the input dictonary.
            if "auxilary_dictionaries" in self.inputs:
                for _, value in self.inputs.auxilary_dictionaries.items():
                    cp2k_utils.dict_merge(input_dict, value)

            if "max_wallclock_seconds" in self.inputs.options[fragment]:
                # Calculation might require up to 5 minutes to gracefully finish the calculation.
                input_dict["GLOBAL"]["WALLTIME"] = max(
                    self.inputs.options[fragment]["max_wallclock_seconds"] - 300, 300
                )

            builder.cp2k.parameters = orm.Dict(dict=input_dict)

            submitted_node = self.submit(builder)
            self.to_context(**{f"scf.{fragment}": submitted_node})

    def run_geo_opts(self):
        for fragment in self.inputs.fragments.keys():
            # We deliberately do not run optimisation for the full structure.
            if fragment == "all":
                continue

            # Generic inputs that are always the same.
            builder = Cp2kBaseWorkChain.get_builder()
            builder.cp2k.code = self.inputs.code
            builder.cp2k.metadata.options = self.inputs.options[fragment]
            builder.cp2k.file = self.ctx.file
            builder.cp2k.metadata.options.parser_name = "cp2k_advanced_parser"

            # Re-loading the input dictionary for the given protocol.
            previous_calc = self.ctx.scf[fragment]
            input_dict = previous_calc.inputs.cp2k.parameters.get_dict()
            input_dict["GLOBAL"]["RUN_TYPE"] = "GEO_OPT"

            # For the geometry optimisation, we reset STRIDE back to 4 4 4.
            input_dict["FORCE_EVAL"]["DFT"]["PRINT"]["E_DENSITY_CUBE"][
                "STRIDE"
            ] = "4 4 4"

            builder.cp2k.parameters = orm.Dict(dict=input_dict)
            builder.cp2k.parent_calc_folder = previous_calc.outputs.remote_folder
            builder.cp2k.structure = previous_calc.inputs.cp2k.structure

            submitted_node = self.submit(builder)
            self.to_context(**{f"opt.{fragment}": submitted_node})

    def finalize(self):
        energies = {}
        self.report("Finalizing...")
        energies["all"] = self.ctx.scf["all"].outputs.output_parameters["energy"]

        separation_energy = energies["all"]
        unrelaxed_separation_energy = energies["all"]

        for fragment in list(self.inputs.fragments.keys()):
            # The geometry optimisation is not run for the full structure.
            if fragment == "all":
                continue

            energies[fragment] = {
                "unrelaxed": self.ctx.scf[fragment].outputs.output_parameters["energy"],
                "relaxed": self.ctx.opt[fragment].outputs.output_parameters["energy"],
            }
            separation_energy -= energies[fragment]["relaxed"]
            unrelaxed_separation_energy -= energies[fragment]["unrelaxed"]

        energies["unrelaxed_separation_energy"] = unrelaxed_separation_energy
        energies["separation_energy"] = separation_energy

        energies = orm.Dict(dict=energies).store()
        self.out("energies", energies)

        # Add the workchain pk to the input structure extras
        common_utils.add_extras(self.inputs.structure, "surfaces", self.node.uuid)
