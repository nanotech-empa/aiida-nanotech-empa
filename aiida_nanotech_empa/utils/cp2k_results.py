"""Read CP2K model and NEB status without modifying provenance or native files."""

import re

PERIODICITIES = ("NONE", "X", "Y", "Z", "XY", "XZ", "YZ", "XYZ")

SOLVER_DIMENSIONS = {
    "PERIODIC": (3,),
    "MT": (0, 2),
    "ANALYTIC": (0, 1, 2),
    "MULTIPOLE": (0,),
    "WAVELET": (0, 2, 3),
    "IMPLICIT": (0, 1, 2, 3),
}


def periodic_pbc(periodic):
    periodic = str(periodic).upper()
    if periodic not in PERIODICITIES:
        raise ValueError(f"Unknown CP2K periodicity: {periodic}")
    return tuple(axis in periodic for axis in "XYZ")


def validate_periodic_solver(periodic, solver):
    dimension = sum(periodic_pbc(periodic))
    if solver not in SOLVER_DIMENSIONS:
        raise ValueError(f"Unknown CP2K Poisson solver: {solver}")
    if dimension not in SOLVER_DIMENSIONS[solver]:
        raise ValueError(
            f"CP2K {solver} does not support {dimension}D periodicity ({periodic}). "
            "MT is implemented only for 0D and 2D; use an explicit supported solver for 1D."
        )
    if solver == "WAVELET" and dimension == 2 and periodic != "XZ":
        raise ValueError("CP2K WAVELET supports 2D periodicity only for XZ")


def calculation_boundary(parameters):
    """Resolve CP2K defaults; reject ambiguous pair-list/electrostatic models."""
    force_eval = parameters.get("FORCE_EVAL", {})
    if isinstance(force_eval, list):
        if len(force_eval) != 1:
            raise ValueError(
                "Cannot infer one boundary model from multiple FORCE_EVAL sections"
            )
        force_eval = force_eval[0]
    periodic = str(
        force_eval.get("SUBSYS", {}).get("CELL", {}).get("PERIODIC", "XYZ")
    ).upper()
    poisson = force_eval.get("DFT", {}).get("POISSON", {})
    electrostatic = str(poisson.get("PERIODIC", "XYZ")).upper()
    pbc = periodic_pbc(periodic)
    periodic_pbc(electrostatic)
    if periodic != electrostatic:
        raise ValueError(
            f"CELL/PERIODIC={periodic} differs from DFT/POISSON/PERIODIC={electrostatic}"
        )
    solver = str(
        poisson.get(
            "POISSON_SOLVER", poisson.get("PSOLVER", poisson.get("POISSON", "PERIODIC"))
        )
    ).upper()
    validate_periodic_solver(periodic, solver)
    return {
        "periodic": periodic,
        "poisson_solver": solver,
        "pbc": pbc,
        "dimension": sum(pbc),
    }


def parse_neb_status(outputs, expected_replicas=None):
    """Separate termination, electronic and band convergence, including replica logs.

    None means that the retained output does not establish convergence. A failed
    late trial SCF does not erase earlier accepted energies/geometries.
    """
    main = outputs.get("aiida.out", "")
    replica_names = sorted(
        (name for name in outputs if re.fullmatch(r"aiida-BAND\d+\.out", name)),
        key=lambda name: int(re.search(r"BAND(\d+)", name)[1]),
    )
    all_text = "\n".join(outputs.values())
    timed_out = "exceeded requested execution time" in all_text
    aborted = "ABORT" in all_text
    scf_failures = all_text.count("SCF run NOT converged")
    scf_timeouts = len(
        re.findall(r"SCF run terminated - exceeded requested execution time", all_text)
    )
    if scf_failures or scf_timeouts or aborted:
        electronic = False
    elif (
        replica_names
        and (expected_replicas is None or len(replica_names) == expected_replicas)
        and all("outer SCF loop converged" in outputs[name] for name in replica_names)
    ):
        electronic = True
    else:
        electronic = None
    checks = []
    for name in ["aiida.out", *replica_names]:
        found = re.findall(
            r"(RMS DISPLACEMENT|MAX DISPLACEMENT|RMS FORCE|MAX FORCE)\s*=\s*"
            r"([\d.Ee+-]+)\s*\[\s*([\d.Ee+-]+)\s*\]\s*\(\s*(YES|NO)\s*\)",
            outputs.get(name, ""),
        )
        if len(found) >= 4:
            checks = [
                {
                    "quantity": q,
                    "value": float(v),
                    "threshold": float(t),
                    "passed": flag == "YES",
                    "file": name,
                }
                for q, v, t, flag in found[-4:]
            ]
    required_checks = {"RMS DISPLACEMENT", "MAX DISPLACEMENT", "RMS FORCE", "MAX FORCE"}
    band = (
        all(c["passed"] for c in checks)
        if {c["quantity"] for c in checks} == required_checks
        else None
    )
    # Termination and convergence are separate facts: a timeout must not
    # overwrite the actual last band criteria (or turn missing evidence into NO).
    replica_scf = {
        name: {
            "converged_calls": outputs[name].count("outer SCF loop converged"),
            "unconverged_warnings": outputs[name].count("SCF run NOT converged"),
            "timeouts": outputs[name].count("exceeded requested execution time"),
        }
        for name in replica_names
    }
    steps = [int(x) for x in re.findall(r"STEP NUMBER\s*=\s*(\d+)", main)]
    warnings = re.findall(r"The number of warnings for this run is\s*:\s*(\d+)", main)
    complete = band is True and electronic is True and not timed_out and not aborted
    return {
        "neb_status_version": 1,
        "exceeded_walltime": timed_out,
        "aborted": aborted,
        "electronic_converged": electronic,
        "band_converged": band,
        "partial_results": bool(steps) and not complete,
        "termination_reason": "aborted"
        if aborted
        else "walltime"
        if timed_out
        else "converged"
        if complete
        else "not_converged"
        if band is False or electronic is False
        else "unknown",
        "nwarnings": int(warnings[-1]) if warnings else scf_failures,
        "scf_unconverged_warnings": scf_failures,
        "scf_timeout_count": scf_timeouts,
        "accepted_band_steps": steps,
        "last_band_convergence": checks,
        "replica_logs_read": replica_names,
        "replica_scf_summary": replica_scf,
    }
