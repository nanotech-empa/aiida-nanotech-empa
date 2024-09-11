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

# Create a methane (CH4) molecule using the ASE molecular database
ch4 = molecule('CH4')

GaussianCalculation = CalculationFactory("gaussian")


def example_dft(gaussian_code):
    """Run a simple gaussian optimization"""

    # structure
    structure = StructureData(ase=ch4)

    num_cores = 1
    memory_mb = 300

    # Main parameters: geometry optimization
    parameters = Dict(
        {
            "link0_parameters": {
                "%chk": "aiida.chk",
                "%mem": "%dMB" % memory_mb,
                "%nprocshared": num_cores,
            },
            "functional": "B3LYP",
            "basis_set": "6-311g(d,p)",
            "charge": 0,
            "multiplicity": 1,
            "dieze_tag": "#P",
            "route_parameters": {
                "scf": {
                    "maxcycle":2048,
                    "cdiis": None,
                    "conver":8
                },
                "int":"superfine",
                "guess":"mix",
                "nosymm": None,
                "opt": "tight",
            },
        }
    )

    # Construct process builder

    builder = GaussianCalculation.get_builder()

    builder.structure = structure
    builder.parameters = parameters
    builder.code = gaussian_code

    builder.metadata.options.resources = {
        "num_machines": 1,
        "tot_num_mpiprocs": num_cores,
    }

    # Should ask for extra +25% extra memory
    builder.metadata.options.max_memory_kb = int(1.25 * memory_mb) * 1024
    builder.metadata.options.max_wallclock_seconds = 5 * 60

    print("Running calculation...")
    res, _node = run_get_node(builder)

    print("Final scf energy: %.4f" % res["output_parameters"]["scfenergies"][-1])


@click.command("cli")
@click.argument("codelabel", default="gaussian@localhost")
def cli(codelabel):
    """Click interface"""
    try:
        code = Code.get_from_string(codelabel)
    except NotExistent:
        print(f"The code '{codelabel}' does not exist")
        sys.exit(1)
    example_dft(code)


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
