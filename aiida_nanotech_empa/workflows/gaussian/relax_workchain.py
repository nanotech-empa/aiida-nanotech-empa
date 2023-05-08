from aiida import engine, orm, plugins

from ...utils import common_utils
from . import common
from .scf_workchain import GaussianScfWorkChain

GaussianBaseWorkChain = plugins.WorkflowFactory("gaussian.base")


class GaussianRelaxWorkChain(engine.WorkChain):
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
            default=lambda: orm.Int(0),
            help="spin multiplicity; 0 means RKS",
        )

        spec.input(
            "wfn_stable_opt",
            valid_type=orm.Bool,
            required=False,
            default=lambda: orm.Bool(False),
            help="if true, perform wfn stability optimization",
        )

        spec.input(
            "tight",
            valid_type=orm.Bool,
            required=False,
            default=lambda: orm.Bool(False),
            help="Use tight optimization criteria.",
        )

        spec.input(
            "freq",
            valid_type=orm.Bool,
            required=False,
            default=lambda: orm.Bool(False),
            help="Also run vibrational analysis.",
        )

        spec.input(
            "empirical_dispersion",
            valid_type=orm.Str,
            required=False,
            default=lambda: orm.Str(""),
            help=("Include empirical dispersion corrections" '(e.g. "GD3", "GD3BJ")'),
        )

        spec.input(
            "constraints",
            valid_type=orm.List,
            required=False,
            default=lambda: orm.List(list=[]),
            help='Supported constraints: ("distance", n1, n2, d)',
        )

        # Do an extra SCF step at the end and potentially create cubes.
        spec.input(
            "basis_set_scf",
            valid_type=orm.Str,
            required=False,
            help="Basis set for SCF. If not present, SCF is skipped.",
        )

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
            help="Use custom metadata.options instead of automatic.",
        )

        spec.outline(
            cls.setup,
            engine.if_(cls.should_do_wfn_stability)(cls.uks_wfn_stability),
            cls.optimization,
            engine.if_(cls.should_do_scf)(cls.scf),
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
            380,
            "ERROR_NO_VIBR_ANALYSIS",
            message="Vibrational analysis did not succeed.",
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

    def should_do_wfn_stability(self):
        if self.is_uks() and self.inputs.wfn_stable_opt:
            return True
        return False

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

        return engine.ExitCode(0)

    def uks_wfn_stability(self):
        self.report("Running UKS WFN Stability")

        parameters = orm.Dict(
            {
                "link0_parameters": self.ctx.link0.copy(),
                "dieze_tag": "#P",
                "functional": self.ctx.functional,
                "basis_set": self.inputs.basis_set.value,
                "charge": 0,
                "multiplicity": self.ctx.mult,
                "route_parameters": {
                    "scf": {
                        "maxcycle": 140,
                    },
                    "Stable": "opt",
                },
            }
        )

        if self.ctx.mult == 1:
            parameters["route_parameters"]["guess"] = "mix"

        builder = GaussianBaseWorkChain.get_builder()

        builder.gaussian.parameters = parameters
        builder.gaussian.structure = self.inputs.structure
        builder.gaussian.code = self.inputs.gaussian_code
        builder.gaussian.metadata.options = self.ctx.metadata_options

        future = self.submit(builder)
        return engine.ToContext(uks_stab=future)

    def optimization(self):
        self.report("Submitting optimization")

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
                    "opt": None,
                },
            }
        )

        builder = GaussianBaseWorkChain.get_builder()

        if self.should_do_wfn_stability():
            parameters["link0_parameters"]["%oldchk"] = "parent_calc/aiida.chk"
            parameters["route_parameters"]["guess"] = "read"
            builder.gaussian.parent_calc_folder = (
                self.ctx.uks_stab.outputs.remote_folder
            )
        elif self.inputs.multiplicity == 1:
            parameters["route_parameters"]["guess"] = "mix"

        # In case of the open-shell singlet, take smaller steps to prevent
        # losing the spin solution.
        if self.inputs.multiplicity == 1:
            parameters["route_parameters"]["opt"] = {"maxstep": 10}

        if self.inputs.freq:
            parameters["route_parameters"]["freq"] = None

        if self.inputs.empirical_dispersion.value != "":
            parameters["route_parameters"][
                "empiricaldispersion"
            ] = self.inputs.empirical_dispersion.value

        opt_dict = {}

        if self.inputs.tight:
            opt_dict["tight"] = None

        if len(self.inputs.constraints) != 0:
            constr_str = ""
            for c in self.inputs.constraints:
                if c[0] == "distance":
                    constr_str += f"{c[1] + 1} {c[2] + 1} ={c[3]:.4f} B\n"
                    constr_str += f"{c[1] + 1} {c[2] + 1} F\n"
                else:
                    self.report(f"Unsupported constraint {c[0]}, skipping.")
            if constr_str != "":
                opt_dict["modredundant"] = None
                parameters["input_parameters"] = {constr_str: None}

        if len(opt_dict) != 0:
            parameters["route_parameters"]["opt"] = opt_dict

        builder.gaussian.parameters = parameters
        builder.gaussian.structure = self.inputs.structure
        builder.gaussian.code = self.inputs.gaussian_code
        builder.gaussian.metadata.options = self.ctx.metadata_options

        submitted_node = self.submit(builder)
        return engine.ToContext(opt=submitted_node)

    def should_do_scf(self):
        return "basis_set_scf" in self.inputs

    def scf(self):
        if not common_utils.check_if_calc_ok(self, self.ctx.opt):
            return self.exit_codes.ERROR_TERMINATION

        self.report("Submitting SCF")

        builder = GaussianScfWorkChain.get_builder()
        builder.gaussian_code = self.inputs.gaussian_code
        builder.structure = self.ctx.opt.outputs.output_structure
        builder.functional = self.inputs.functional
        builder.empirical_dispersion = self.inputs.empirical_dispersion
        builder.basis_set = self.inputs.basis_set_scf
        builder.multiplicity = self.inputs.multiplicity
        builder.parent_calc_folder = self.ctx.opt.outputs.remote_folder

        if "formchk_code" in self.inputs and "cubegen_code" in self.inputs:
            builder.formchk_code = self.inputs.formchk_code
            builder.cubegen_code = self.inputs.cubegen_code
            builder.cubes_n_occ = self.inputs.cubes_n_occ
            builder.cubes_n_virt = self.inputs.cubes_n_virt
            builder.cubes_edge_space = self.inputs.cubes_edge_space
            builder.cubegen_parser_name = self.inputs.cubegen_parser_name
            builder.cubegen_parser_params = self.inputs.cubegen_parser_params

        if "options" in self.inputs:
            builder.options = self.inputs.options

        submitted_node = self.submit(builder)
        return engine.ToContext(scf=submitted_node)

    def finalize(self):
        if not common_utils.check_if_calc_ok(self, self.ctx.opt):
            return self.exit_codes.ERROR_TERMINATION

        if self.inputs.freq:
            if "vibfreqs" not in dict(self.ctx.opt.outputs.output_parameters):
                return self.exit_codes.ERROR_NO_VIBR_ANALYSIS

        self.report("Finalizing...")

        self.out("output_structure", self.ctx.opt.outputs.output_structure)
        self.out("energy_ev", self.ctx.opt.outputs.energy_ev)
        self.out("output_parameters", self.ctx.opt.outputs.output_parameters)
        self.out("remote_folder", self.ctx.opt.outputs.remote_folder)

        if "scf" in self.ctx:
            if not common_utils.check_if_calc_ok(self, self.ctx.scf):
                return self.exit_codes.ERROR_TERMINATION
            self.out("scf_energy_ev", self.ctx.scf.outputs.energy_ev)
            self.out("scf_output_parameters", self.ctx.scf.outputs.output_parameters)
            self.out("scf_remote_folder", self.ctx.scf.outputs.remote_folder)
            for cubes_out in list(self.ctx.scf.outputs):
                if cubes_out.startswith("cube"):
                    self.out(cubes_out, self.ctx.scf.outputs[cubes_out])
                elif cubes_out == "retrieved":
                    self.out("cubes_retrieved", self.ctx.scf.outputs[cubes_out])

        return engine.ExitCode(0)
