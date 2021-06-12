#!/bin/bash

sudo apt install build-essential -y
sudo apt-get install gfortran -y

wget https://github.com/QEF/q-e/releases/download/qe-6.7.0/qe-6.7-ReleasePack.tgz

tar xvf qe-6.7-ReleasePack.tgz

rm qe-6.7-ReleasePack.tgz

mv qe-6.7 qe

cd qe

./configure --disable-parallel

make pw pp
