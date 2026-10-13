#!/usr/bin/env bash
set -euo pipefail

# Reviewed Bader 1.05 source, also used for the local Surfaces preview codes.
revision=fadab8cb92fe947b0fd8aa473cb5aaf909073983
mkdir -p bader
curl --fail --location --retry 3 \
    "https://github.com/nanotech-empa/aiidalab-alps-files/archive/${revision}.tar.gz" \
    | tar -xz --strip-components=2 -C bader \
        "aiidalab-alps-files-${revision}/bader_src"
make -C bader -f makefile.lnx_ifort FC="${FC:-gfortran}" bader
test -x bader/bader
