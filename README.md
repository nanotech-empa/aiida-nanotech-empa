[![Build Status](https://github.com/nanotech-empa/aiida-nanotech-empa/workflows/ci/badge.svg?branch=master)](https://github.com/nanotech-empa/aiida-nanotech-empa/actions)
[![codecov](https://codecov.io/gh/nanotech-empa/aiida-nanotech-empa/branch/develop/graph/badge.svg?token=52ACMY55UQ)](https://codecov.io/gh/nanotech-empa/aiida-nanotech-empa)
[![PyPI version](https://badge.fury.io/py/aiida-nanotech-empa.svg)](https://badge.fury.io/py/aiida-nanotech-empa)
[![DOI](https://zenodo.org/badge/275159349.svg)](https://zenodo.org/badge/latestdoi/275159349)

# aiida-nanotech-empa

AiiDA library containing plugins, parsers, and workflows developed by the nanotech@surfaces group at Empa.

The package focuses on automation around CP2K, Gaussian, Quantum ESPRESSO, and post-processing tools used in surface-science workflows. It is also used by the `aiidalab-empa-surfaces` app.

## Contents

### CP2K workflows

* `nanotech_empa.cp2k.geo_opt`: CP2K geometry and cell optimization workflows with restart handling and optional cube post-processing.
* `nanotech_empa.cp2k.scf`: single-point CP2K SCF workflow. It can run an OT-only energy calculation, an optional diagonalization step for empty states and AO overlap matrices, sparse AO-overlap retrieval, or Bader charge analysis from a fine charge-density cube.
* `nanotech_empa.cp2k.diag`: OT plus diagonalization SCF workflow used by STM/PDOS-style workflows and by the SCF workflow.
* `nanotech_empa.cp2k.stm`, `nanotech_empa.cp2k.afm`, `nanotech_empa.cp2k.hrstm`, `nanotech_empa.cp2k.orbitals`, `nanotech_empa.cp2k.pdos`: workflows for scanning-probe simulations, orbital cubes, and projected density of states.
* `nanotech_empa.cp2k.fragment_separation`, `nanotech_empa.cp2k.replica`, `nanotech_empa.cp2k.neb`, `nanotech_empa.cp2k.phonons`: workflows for adsorption energies, replica chains, nudged elastic band calculations, and phonons.
* `nanotech_empa.cp2k.ads_gw_ic`, `nanotech_empa.cp2k.molecule_gw`, `nanotech_empa.cp2k.mol_opt_gw`: CP2K/GW-oriented workflows.

### Calculation plugins

* `nanotech_empa.bader`: runs Bader charge analysis and retrieves `ACF.dat`, `AVF.dat`, and `BCF.dat`.
* `nanotech_empa.sparse_overlap`: converts CP2K AO overlap matrix logs to sparse `.npz` files using an external converter executable.
* `nanotech_empa.stm`, `nanotech_empa.overlap`, `nanotech_empa.afm`, `nanotech_empa.hrstm`, `nanotech_empa.cubehandler`: post-processing plugins used by the CP2K workflows.

### Gaussian and Quantum ESPRESSO workflows

* `nanotech_empa.gaussian.*`: Gaussian workflows for SCF, relaxations, spin-state analysis, constrained optimization, CASSCF, NICS, and related post-processing.
* `nanotech_empa.nanoribbon`: workflow to characterize 1D periodic systems with Quantum ESPRESSO.

## Installation

```shell
pip install aiida-nanotech-empa
```

## For maintainers

To create a new release, clone the repository, install development dependencies with `pip install '.[dev]'`, and then execute `bumpver update --dry --major (--minor/--patch)`.
This will display the changes that will be made to the repository - check them carefully.

Once you are happy with the changes, remove the `--dry` option and re-execute the command.
This will:

  1. Create a tagged release with bumped version and push it to the repository.
  2. Trigger a GitHub actions workflow that creates a GitHub release.



Additional notes:

  - The release tag (e.g. a/b/rc) is determined from the last release.
    Use the `--tag beta (alpha/gamma)`  option to switch the release tag.

## Acknowledgements
We acknowledge support from:
* the [NCCR MARVEL](http://nccr-marvel.ch/) funded by the Swiss National Science Foundation;
* the [swissuniversities P-5 project "Materials Cloud"](https://www.materialscloud.org/swissuniversities).
<img src="https://github.com/nanotech-empa/aiida-nanotech-empa/blob/master/images/MARVEL.png" width="250px" height="131px"/>
<img src="https://github.com/nanotech-empa/aiida-nanotech-empa/blob/master/images/swissuniversities.png" width="300px" height="35px"/>
