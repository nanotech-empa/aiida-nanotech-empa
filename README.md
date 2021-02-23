[![Build Status](https://github.com/nanotech-empa/aiida-nanotech-empa/workflows/ci/badge.svg?branch=master)](https://github.com/nanotech-empa/aiida-nanotech-empa/actions)
[![PyPI version](https://badge.fury.io/py/aiida-nanotech-empa.svg)](https://badge.fury.io/py/aiida-nanotech-empa)

# aiida-nanotech-empa

AiiDA library containing plugins/workflows developed at nanotech@surfaces group from Empa.

Contents:

* `nanotech_empa.nanoribbon`: work chain to characterize 1D periodic systems based on Quantum Espresso

* `nanotech_empa.gaussian.spin`: Work chain to characterize spin properties of molecular systems with Gaussian. Calls multiple child work chains. Steps:
  * Wavefunction stability is tested for each spin multiplicity
  * Different spin states are optimized and ground state is found
  * Ground state properties are calculated: ionization potential and electron affinity with Δ-SCF, natural orbital analysis in case of open-shell singlet
  * Vertical excitation energies for other multiplicities
  * Orbitals and densities are rendered with PyMOL (needs to be installed separately)

## Installation

```shell
pip install aiida-nanotech-empa
```
