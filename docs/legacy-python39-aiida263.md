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
| aiidalab-eln | [v0.1.5](https://github.com/aiidalab/aiidalab-eln/releases/tag/v0.1.5) |
| aiidalab-openbis | [v0.1.2](https://github.com/nanotech-empa/aiidalab-openbis/releases/tag/v0.1.2) |
| aiidalab-empa-nanoribbons | [v2.2.0](https://github.com/nanotech-empa/aiidalab-empa-nanoribbons/releases/tag/v2.2.0) |

The backend, Surfaces, SPM tools and Nanoribbons use their existing immutable
release tags. Setup preserves its existing 2025.0214 source version. openBIS
0.1.2 adds Fabio's synchronization, measurement-upload, logging and
machine-configuration fixes through nanotech-empa/aiidalab-openbis#151.
Existing older tags remain available.

ELN 0.1.5 is published from Fabio's approved
[aiidalab/aiidalab-eln#95](https://github.com/aiidalab/aiidalab-eln/pull/95),
merged at `729108927ff72cf7202dbb1b9cc69b3bf626e712`. It includes the
configuration/shared-widget fixes, Fabio's scikit-learn dependency correction
and the missing widgets-base dependency. All seven legacy source selections
above are now published. Existing older tags remain available.

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

The original seven source selections and their declared Python/AiiDA
requirements were checked on 2026-10-27. The openBIS 0.1.2 release preparation
on 2026-11-06 additionally builds and installs both the openBIS wheel and the
ELN 0.1.5 candidate in a clean Python 3.9.25 / AiiDA 2.6.3 runner. The combined
checks passed: wheel installation, imports outside the checkout, `pip check`,
4 ELN tests and 39 openBIS regressions. The resolved environment includes
widgets-base 2.5.1, ipywidgets 7.7.4 and pyBIS 7.0.0rc6. The checks use
independent AiiDA state and mocked external services. The ELN
candidate was commit `2e09cd578bf7b4aa6cfa068488f26c9ed89eeaa9`. Published
ELN v0.1.5 has the same source tree. On 2026-11-12 the official publication
workflow completed successfully; its GitHub and PyPI wheel/sdist checksums
matched, and both artifacts' package source matched the reviewed tree.
Validation results and the resolved package inventory are linked from the
openBIS release notes. These publication checks do not expand the earlier
legacy runtime qualification.

The primary-package constraints are not a complete dependency lock. Keep the
existing tested AiiDAlab, ipywidgets, plugin dependencies, container image,
external-code versions and openBIS configuration with a legacy deployment.
The automated checks do not establish browser/live-server acceptance or a
production deployment. Source tags do not freeze those components or change
an AiiDA database. Do not use these package pins to downgrade a database
already upgraded by a newer AiiDA.

ELN 0.1.5 includes [ELN #94](https://github.com/aiidalab/aiidalab-eln/pull/94)
(scikit-learn dependency) and the configuration/shared-widget changes merged
since 0.1.4. openBIS 0.1.2 retains the rooms, chatbot, project-dropdown and
synchronization features from 0.1.1 and adds the fixes in
[#145](https://github.com/nanotech-empa/aiidalab-openbis/pull/145) through
[#151](https://github.com/nanotech-empa/aiidalab-openbis/pull/151).
Its package-discovery correction also includes the `aiida_openbis` and
`nanonis_importer` packages and their required package resources in the wheel.
For the complete changes, compare
[ELN release preparation](https://github.com/aiidalab/aiidalab-eln/pull/95) and
[openBIS v0.1.1...v0.1.2](https://github.com/nanotech-empa/aiidalab-openbis/compare/v0.1.1...v0.1.2).
