#GCC=gcc-6.2.0
#GCC=gcc-5.3.0
GCC=gcc-4.8.2

#export PATH=/home/develop/Python-2.7/bin:/usr/local/$GCC/bin:$PATH
export PATH=/usr/local/$GCC/bin:$PATH
export CC=/usr/local/$GCC/bin/gcc
export CXX=/usr/local/$GCC/bin/g++

source /opt/intel/mkl/bin/mklvars.sh intel64
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/develop/Python-2.7/lib:/usr/local/$GCC/lib64:/usr/local/VTK-7.0.0/lib:$HOME/Downloads/paralution-1.1.0/build/lib:/usr/local/Qt-4.8.7/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/Python-2.7/lib:/usr/local/$GCC/lib64:/usr/local/VTK-7.0.0/lib:$PWD/paralution-1.1.0/build/lib:/usr/local/Qt-4.8.7/lib
export OMP_NUM_THREADS=4

