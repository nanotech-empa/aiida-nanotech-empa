#!/bin/bash
set -e

sudo apt-get update
sudo apt-get install -y mpich libmpich-dev

pip install cp2k-spm-tools

which cp2k-stm-sts-wfn
which cp2k-overlap-from-wfns
