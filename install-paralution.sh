#!/bin/bash
set -ev
wget http://www.paralution.com/downloads/paralution-1.1.0.tar.gz
tar xzf paralution-1.1.0.tar.gz
patch -p0 < paralution.patch
cd paralution-1.1.0
mkdir build
cd build
cmake ..
make -j2
