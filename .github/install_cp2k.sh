#!/bin/bash

mkdir cp2k
cd cp2k

wget https://github.com/cp2k/cp2k/releases/download/v8.1.0/cp2k-8.1-Linux-x86_64.ssmp

chmod +x cp2k-8.1-Linux-x86_64.ssmp

ln -s cp2k-8.1-Linux-x86_64.ssmp cp2k.ssmp
