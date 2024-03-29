[![Build Status](https://github.com/nanotech-empa/aiida-nanotech-empa/workflows/ci/badge.svg?branch=master)](https://github.com/nanotech-empa/aiida-nanotech-empa/actions)
[![codecov](https://codecov.io/gh/nanotech-empa/aiida-nanotech-empa/branch/develop/graph/badge.svg?token=52ACMY55UQ)](https://codecov.io/gh/nanotech-empa/aiida-nanotech-empa)
[![PyPI version](https://badge.fury.io/py/aiida-nanotech-empa.svg)](https://badge.fury.io/py/aiida-nanotech-empa)
[![DOI](https://zenodo.org/badge/275159349.svg)](https://zenodo.org/badge/latestdoi/275159349)

# aiida-nanotech-empa

AiiDA library containing plugins/workflows developed at nanotech@surfaces group from Empa.

Contents:

* `nanotech_empa.nanoribbon`: work chain to characterize 1D periodic systems based on Quantum Espresso.

* `nanotech_empa.gaussian.spin`: Work chain to characterize spin properties of molecular systems with Gaussian. Calls multiple child work chains. Steps:
  * Wavefunction stability is tested for each spin multiplicity
  * Geometry is relaxed for the different spin states and ground state is found
  * Property calcuation on the ground state: ionization potential and electron affinity with Δ-SCF, natural orbital analysis in case of open-shell singlet
  * Vertical excitation energies for non-ground state multiplicities
  * Orbitals and densities are rendered with PyMOL (needs to be installed separately as a python library, e.g. from [pymol-open-source](https://github.com/schrodinger/pymol-open-source/tree/v2.4.0))

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
