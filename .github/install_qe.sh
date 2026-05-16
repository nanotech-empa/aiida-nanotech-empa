#!/bin/bash
set -e

sudo apt-get update
sudo apt-get install -y quantum-espresso

which pw.x
which pp.x
which projwfc.x
