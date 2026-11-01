# Bundled CP2K UZH parameter data

The following files come from the CP2K project, release **2024.1**, commit
[`077bf32593c66d7a8606997006b97a99204d7b92`](https://github.com/cp2k/cp2k/tree/077bf32593c66d7a8606997006b97a99204d7b92/data):

- `BASIS_MOLOPT_UZH`: UZH orbital basis definitions.
- `BASIS_ADMM_UZH`: UZH auxiliary basis definitions for ADMM.
- `POTENTIAL_UZH`: matching UZH pseudopotential definitions.
- `t_c_g.dat`: truncated-Coulomb integral data.

Their authorship and existing notices belong to the upstream CP2K contributors.
The upstream licence is preserved in [CP2K-LICENSE](CP2K-LICENSE); these third-party
files are not relicensed under this package's MIT licence. Preserve the upstream
notices and licence when redistributing them.

The three basis/potential files retain the upstream numerical content and comments;
only trailing whitespace and a final blank line were normalized. `t_c_g.dat`
matches the upstream file byte for byte. No numerical data were changed during
this review.

The upstream SHA-256 checksums before whitespace normalization are:

| File | SHA-256 |
| --- | --- |
| `BASIS_MOLOPT_UZH` | `394bf1196353838cec389cb079a41a81b42a07afa49a18ededb9cc6e053c0291` |
| `BASIS_ADMM_UZH` | `3f08699fd16a79e116de65f7520db362f71468657ab5e1c90c735fa4930393c1` |
| `POTENTIAL_UZH` | `7fd6942a627c7d1a897d938e198ea3c930630ac98d3e7099426bd83596c6b763` |
| `t_c_g.dat` | `8d01f627193427a90ae6ed7821c24acadce60b3f560369a7bd07107e76ad9919` |

## Default PBE0 element mappings

Each default in `atomic_kinds.yml` must name an orbital basis, an `AUX_FIT`
basis and a pseudopotential that are present in these files and have matching
valence-electron counts. Na uses q9, Mg q10 and Rh q17 in all three mappings;
these are the matched sets supplied by this snapshot and include semicore
electrons explicitly.

The bundled UZH orbital and ADMM files have no lanthanide entries (La through Lu).
No default PBE0 sets are advertised for those elements. A custom setup requires
explicit compatible overrides and corresponding data files. Unsupported defaults
fail during input construction with a descriptive error.

The ordinary PBE defaults are unchanged. These files provide the data for the
PBE0 workflow series; diagonalization, dispersion and other workflow settings
are reviewed in their own components. Availability of an entry does not establish
basis convergence for a particular scientific calculation.
