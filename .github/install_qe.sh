#!/bin/bash

sudo apt-get install gfortran -y

wget https://github.com/QEF/q-e/releases/download/qe-6.7.0/qe-6.7-ReleasePack.tgz

tar xvf qe-6.7-ReleasePack.tgz

rm qe-6.7-ReleasePack.tgz

cd qe-6.7

./configure --disable-parallel

make pw pp

echo "export \"PATH=`pwd`/bin:\$PATH\"" >> ~/.profile
source ~/.profile