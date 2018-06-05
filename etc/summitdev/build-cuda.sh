#!/bin/bash

module load cmake cuda

BUILD_DIR=$(pwd)

if [ ! -d hypre ] ; then
    git clone git@github.com:exawind/hypre.git
fi

### Compile hypre
cd hypre/src
CUDACXX=mpixlC CC=nvcc CXX=nvcc ./configure --with-cuda --enable-unified-memory --without-superlu --prefix=${BUILD_DIR}/install
make clean
make -j 12
make install


cd ${BUILD_DIR}
CC=mpixlc CXX=mpixlC cmake -DHYPRE_DIR=${BUILD_DIR}/install ../
cp link_working.txt src/CMakeFiles/hypre_app.dir/link.txt
make -j 12
