# Empa legacy stack: Python 3.9 / AiiDA 2.6.3

These source versions preserve the legacy application line while the modern
applications move to Python >=3.12 and AiiDA >=2.8. Keep an existing legacy
installation on Python 3.9.x and `aiida-core==2.6.3`, and select the tagged
sources below instead of a moving `main` or `master` branch.

| Repository | Legacy source version |
| --- | --- |
| aiida-nanotech-empa | [v1.0.0](https://github.com/nanotech-empa/aiida-nanotech-empa/releases/tag/v1.0.0) |
| aiidalab-empa-surfaces | [v1.1.0](https://github.com/nanotech-empa/aiidalab-empa-surfaces/releases/tag/v1.1.0) |
| cp2k-spm-tools | [v1.5.0](https://github.com/nanotech-empa/cp2k-spm-tools/releases/tag/v1.5.0) |
| aiidalab-empa-setup | [v2025.0214](https://github.com/nanotech-empa/aiidalab-empa-setup/releases/tag/v2025.0214) |
| aiidalab-eln | [v0.1.4](https://github.com/aiidalab/aiidalab-eln/releases/tag/v0.1.4) |
| aiidalab-openbis | [v0.1.1](https://github.com/nanotech-empa/aiidalab-openbis/releases/tag/v0.1.1) |
| aiidalab-empa-nanoribbons | [v2.2.0](https://github.com/nanotech-empa/aiidalab-empa-nanoribbons/releases/tag/v2.2.0) |

The backend, Surfaces, SPM tools and Nanoribbons use their existing immutable
release tags. Setup preserves its existing 2025.0214 source version. openBIS 0.1.1 preserves
the latest merged legacy code, including Fabio's changes, with version metadata
and documentation updates only. Existing older tags remain available.

ELN 0.1.5 is prepared in [aiidalab/aiidalab-eln#95](https://github.com/aiidalab/aiidalab-eln/pull/95)
to preserve its latest merged code, including Fabio's scikit-learn dependency
fix. It requires review before release; the published ELN 0.1.4 remains pinned
here until then. Do not replace that pin with an unpublished version.

## Selecting an application

The [primary-package constraints](../constraints/legacy-python39-aiida263.txt)
pin the versions above together with AiiDA 2.6.3. Download that file and use it
when installing a selected application from its tag. For example, from a
Python 3.9 legacy environment:

```sh
python -m pip install -c legacy-python39-aiida263.txt \
  "git+https://github.com/nanotech-empa/aiidalab-empa-surfaces.git@v1.1.0"
```

Constraints restrict versions; they do not install every listed application.
SPM tools may run in a separate post-processing environment on a compute
system; use v1.5.0 there when retaining Python 3.9. SPM tools and the setup
app do not themselves declare an AiiDA dependency.

## Validation and preserved environment

On 2026-10-27, the selected source identities and declared Python/AiiDA
requirements were checked. The tracked Python files in the selected
source snapshots and the proposed ELN 0.1.5 source parse with Python 3.9
syntax rules. ELN main
and openBIS master still declare requirements admitting this legacy runtime;
their modern integration PRs are separate.

This is a source selection and primary-package constraint set, not a complete
dependency lock or a new Python 3.9 installation/browser validation. Keep the
existing tested AiiDAlab, ipywidgets, plugin dependencies, container image,
external-code versions and openBIS configuration with a legacy deployment.
The source tags do not freeze those components or change an AiiDA database.
Do not use these package pins to downgrade a database already upgraded by a
newer AiiDA.

The proposed ELN 0.1.5 includes [ELN #94](https://github.com/aiidalab/aiidalab-eln/pull/94)
(scikit-learn dependency). openBIS 0.1.1 includes
[#139](https://github.com/nanotech-empa/aiidalab-openbis/pull/139) (rooms before instruments),
[#140](https://github.com/nanotech-empa/aiidalab-openbis/pull/140) (chatbot interface),
[#141](https://github.com/nanotech-empa/aiidalab-openbis/pull/141) (project dropdown) and
[#143](https://github.com/nanotech-empa/aiidalab-openbis/pull/143) (synchronisation interface).
For the complete changes, compare
[ELN v0.1.4...v0.1.5](https://github.com/aiidalab/aiidalab-eln/compare/v0.1.4...v0.1.5) and
[openBIS v0.1.0...v0.1.1](https://github.com/nanotech-empa/aiidalab-openbis/compare/v0.1.0...v0.1.1).
