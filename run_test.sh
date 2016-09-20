#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/paralution-1.1.0/build/lib
echo $LD_LIBRARY_PATH
$1 csr_test.py
