[![Build Status](https://github.com/nanotech-empa/aiida-nanotech-empa/workflows/ci/badge.svg?branch=master)](https://github.com/nanotech-empa/aiida-nanotech-empa/actions)
[![codecov](https://codecov.io/gh/nanotech-empa/aiida-nanotech-empa/branch/develop/graph/badge.svg?token=52ACMY55UQ)](https://codecov.io/gh/nanotech-empa/aiida-nanotech-empa)
[![PyPI version](https://badge.fury.io/py/aiida-nanotech-empa.svg)](https://badge.fury.io/py/aiida-nanotech-empa)

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
