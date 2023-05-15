from aiida import engine, orm, plugins

from ...utils import common_utils
from . import common

GaussianBaseWorkChain = plugins.WorkflowFactory("gaussian.base")
GaussianCubesWorkChain = plugins.WorkflowFactory("gaussian.cubes")


class GaussianScfWorkChain(engine.WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("gaussian_code", valid_type=orm.Code)
        spec.input(
            "structure",
            valid_type=orm.StructureData,
            required=True,
            help="input geometry",
        )
        spec.input(
            "functional", valid_type=orm.Str, required=True, help="xc functional"
        )
        spec.input("basis_set", valid_type=orm.Str, required=True, help="basis_set")
        spec.input(
            "multiplicity",
            valid_type=orm.Int,
            required=False,
            default=lambda: orm.Int(1),
            help="Spin multiplicity; 0 means RKS",
        )
        spec.input(
            "wfn_stable_opt",
            valid_type=orm.Bool,
            required=False,
            default=lambda: orm.Bool(False),
            help="If true, perform wfn stability optimization.",
        )

        spec.input(
            "wfn_stable_opt_min_basis",
            valid_type=orm.Bool,
            required=False,
            default=lambda: orm.Bool(False),
            help="If true, perform first a minimal basis stability opt.",
        )

        spec.input(
            "empirical_dispersion",
            valid_type=orm.Str,
            required=False,
            default=lambda: orm.Str(""),
            help=("Include empirical dispersion corrections" '(e.g. "GD3", "GD3BJ")'),
        )

        spec.input(
            "parent_calc_folder",
            valid_type=orm.RemoteData,
            required=False,
            help="the folder of a completed gaussian calculation",
        )

        # Inputs for cubes generation.
        spec.input("formchk_code", valid_type=orm.Code, required=False)
        spec.input("cubegen_code", valid_type=orm.Code, required=False)

        spec.input(
            "cubes_n_occ",
            valid_type=orm.Int,
            required=False,
            default=lambda: orm.Int(0),
            help="Number of occupied orbital cubes to generate",
        )

        spec.input(
            "cubes_n_virt",
            valid_type=orm.Int,
            required=False,
            default=lambda: orm.Int(0),
            help="Number of virtual orbital cubes to generate",
        )

        spec.input(
            "cubes_edge_space",
            valid_type=orm.Float,
            required=False,
            default=lambda: orm.Float(3.0),
            help="Extra cube space in addition to bounding box [ang].",
        )

        spec.input(
            "cubegen_parser_name",
            valid_type=str,
            default="nanotech_empa.gaussian.cubegen_pymol",
            non_db=True,
        )

        spec.input(
            "cubegen_parser_params",
            valid_type=orm.Dict,
            required=False,
            default=lambda: orm.Dict(dict={}),
            help="Additional parameters to cubegen parser.",
        )

        spec.input(
            "options",
            valid_type=orm.Dict,
            required=False,
            help="Use custom metadata.options instead of the automatic ones.",
        )

        spec.outline(
            cls.setup,
            engine.if_(cls.should_do_min_basis_stable_opt)(cls.min_basis_stable_opt),
            cls.scf,
            engine.if_(cls.did_scf_fail)(cls.scf),
            engine.if_(cls.should_do_cubes)(cls.cubes),
            cls.finalize,
        )

        spec.outputs.dynamic = True

        spec.exit_code(
            301,
            "ERROR_MULTIPLICITY",
            message="Multiplicity and number of el. doesn't match.",
        )
        spec.exit_code(
            302,
            "ERROR_OPTIONS",
            message="Input options are invalid.",
        )
        spec.exit_code(
            390,
            "ERROR_TERMINATION",
            message="One or more steps of the work chain failed.",
        )

    def is_uks(self):
        if self.inputs.multiplicity.value == 0:
            return False
        return True

    def setup(self):
        self.report("Inspecting input and setting up things")

        common.setup_context_variables(self)

        if self.ctx.mult % 2 == self.ctx.n_electrons % 2:
            return self.exit_codes.ERROR_MULTIPLICITY

        success = common.determine_metadata_options(self)
        if not success:
            return self.exit_codes.ERROR_OPTIONS

        num_cores, memory_mb = common.get_gaussian_cores_and_memory(
            self.ctx.metadata_options, self.ctx.comp
        )

        self.ctx.link0 = {
            "%chk": "aiida.chk",
            "%mem": "%dMB" % memory_mb,
            "%nprocshared": str(num_cores),
        }

        # Use default convergence criterion at the start
        # but switch to conver=7 in case of failure.
        self.ctx.conver = None
        self.ctx.scf_label = "scf"

        return engine.ExitCode(0)

    def should_do_min_basis_stable_opt(self):
        return self.inputs.wfn_stable_opt_min_basis

    def min_basis_stable_opt(self):
        self.report("Submitting minimal basis stable opt")

        parameters = orm.Dict(
            {
                "link0_parameters": self.ctx.link0.copy(),
                "dieze_tag": "#P",
                "functional": self.ctx.functional,
                "basis_set": "STO-3G",
                "charge": 0,
                "multiplicity": self.ctx.mult,
                "route_parameters": {
                    "scf": {"maxcycle": 140},
                    "stable": "opt",
                },
            }
        )

        builder = GaussianBaseWorkChain.get_builder()

        if "parent_calc_folder" in self.inputs:
            # Read WFN from parent calc
            parameters["link0_parameters"]["%oldchk"] = "parent_calc/aiida.chk"
            parameters["route_parameters"]["guess"] = "read"
            builder.gaussian.parent_calc_folder = self.inputs.parent_calc_folder
        elif self.is_uks() and self.ctx.mult == 1:
            # For open-shell singlet, mix homo & lumo
            parameters["route_parameters"]["guess"] = "mix"

        builder.gaussian.parameters = parameters
        builder.gaussian.structure = self.inputs.structure
        builder.gaussian.code = self.inputs.gaussian_code
        builder.gaussian.metadata.options = self.ctx.metadata_options

        future = self.submit(builder)
        return engine.ToContext(min_stable=future)

    def scf(self):
        self.report("Submitting SCF")

        parameters = orm.Dict(
            {
                "link0_parameters": self.ctx.link0.copy(),
                "dieze_tag": "#P",
                "functional": self.ctx.functional,
                "basis_set": self.inputs.basis_set.value,
                "charge": 0,
                "multiplicity": self.ctx.mult,
                "route_parameters": {
                    "scf": {"maxcycle": 140},
                },
            }
        )
        if self.inputs.wfn_stable_opt:
            parameters["route_parameters"]["stable"] = "opt"
        else:
            parameters["route_parameters"]["sp"] = None

        builder = GaussianBaseWorkChain.get_builder()

        if self.should_do_min_basis_stable_opt():
            # Read WFN from min basis stable opt
            parameters["link0_parameters"]["%oldchk"] = "parent_calc/aiida.chk"
            parameters["route_parameters"]["guess"] = "read"
            builder.gaussian.parent_calc_folder = (
                self.ctx.min_stable.outputs.remote_folder
            )
        elif "parent_calc_folder" in self.inputs:
            # Read WFN from parent calc
            parameters["link0_parameters"]["%oldchk"] = "parent_calc/aiida.chk"
            parameters["route_parameters"]["guess"] = "read"
            builder.gaussian.parent_calc_folder = self.inputs.parent_calc_folder
        elif self.is_uks() and self.ctx.mult == 1:
            # For open-shell singlet, mix homo & lumo
            parameters["route_parameters"]["guess"] = "mix"

        if self.inputs.empirical_dispersion.value != "":
            parameters["route_parameters"][
                "empiricaldispersion"
            ] = self.inputs.empirical_dispersion.value

        if self.ctx.conver is not None:
            parameters["route_parameters"]["scf"]["conver"] = self.ctx.conver

        builder.gaussian.parameters = parameters
        builder.gaussian.structure = self.inputs.structure
        builder.gaussian.code = self.inputs.gaussian_code
        builder.gaussian.metadata.options = self.ctx.metadata_options

        future = self.submit(builder)
        future.description = self.ctx.scf_label
        return engine.ToContext(**{self.ctx.scf_label: future})

    def did_scf_fail(self):
        scf_node = self.ctx[self.ctx.scf_label]
        if not common_utils.check_if_calc_ok(self, scf_node):
            # Set up for conver=7 calculation.
            self.report("SCF failed with default convergence criterion!")
            self.report("Switching to a looser threshold.")
            self.ctx.conver = 7
            self.ctx.scf_label = f"scf_c{self.ctx.conver}"
            return True
        return False

    def should_do_cubes(self):
        codes_set = "formchk_code" in self.inputs and "cubegen_code" in self.inputs
        non_zero_num = (
            self.inputs.cubes_n_occ.value > 0 and self.inputs.cubes_n_virt.value > 0
        )
        return codes_set and non_zero_num

    def cubes(self):
        scf_node = self.ctx[self.ctx.scf_label]

        if not common_utils.check_if_calc_ok(self, scf_node):
            return self.exit_codes.ERROR_TERMINATION

        self.report("Generating cubes")

        orb_index_list = list(
            range(
                -self.inputs.cubes_n_occ.value + 1, self.inputs.cubes_n_virt.value + 1
            )
        )

        future = self.submit(
            GaussianCubesWorkChain,
            formchk_code=self.inputs.formchk_code,
            cubegen_code=self.inputs.cubegen_code,
            gaussian_calc_folder=scf_node.outputs.remote_folder,
            gaussian_output_params=scf_node.outputs["output_parameters"],
            orbital_indexes=orm.List(list=orb_index_list),
            orbital_index_ref=orm.Str("half_num_el"),
            edge_space=self.inputs.cubes_edge_space,
            dx=orm.Float(0.15),
            retrieve_cubes=orm.Bool(False),
            cubegen_parser_name=self.inputs.cubegen_parser_name,
            cubegen_parser_params=self.inputs.cubegen_parser_params,
        )
        return engine.ToContext(cubes=future)

    def finalize(self):
        scf_node = self.ctx[self.ctx.scf_label]

        if not common_utils.check_if_calc_ok(self, scf_node):
            return self.exit_codes.ERROR_TERMINATION

        self.report("Finalizing...")

        if self.should_do_cubes():
            if not common_utils.check_if_calc_ok(self, self.ctx.cubes):
                return self.exit_codes.ERROR_TERMINATION
            for cubes_out in list(self.ctx.cubes.outputs):
                if cubes_out.startswith("cube"):
                    self.out(cubes_out, self.ctx.cubes.outputs[cubes_out])
                elif cubes_out == "retrieved":
                    self.out("cubes_retrieved", self.ctx.cubes.outputs[cubes_out])

        self.out("energy_ev", scf_node.outputs.energy_ev)
        self.out("output_parameters", scf_node.outputs["output_parameters"])

        self.out("remote_folder", scf_node.outputs["remote_folder"])

        return engine.ExitCode(0)
