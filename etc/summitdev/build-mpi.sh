#!/bin/bash

module load cuda cmake

BUILD_DIR=$(pwd)

if [ ! -d hypre ] ; then
    git clone git@github.com:exawind/hypre.git
fi

### Compile hypre
cd hypre/src
CC=mpixlc CXX=mpixlC ./configure --without-openmp --without-superlu --prefix=${BUILD_DIR}/install
make clean
make -j 12
make install


cd ${BUILD_DIR}
CC=mpixlc CXX=mpixlC -DHYPRE_DIR=${BUILD_DIR}/install ../
make -j 12
