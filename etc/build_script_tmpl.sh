#!/bin/bash
#
# HYPRE mini-app build script template
#
# Copy, edit, and execute from build directory
#

###
### ACTIVATE NECESSARY MODULES HERE
###

BUILD_DIR=$(pwd)
BASE_DIR=${BUILD_DIR}/..
HYPRE_DIR=${BASE_DIR}/deps/hypre/src
HYPRE_INSTALL_DIR=${BUILD_DIR}/hypre_install
NPROCS=${NPROCS:-8}

# Build and install HYPRE first
pushd ${HYPRE_DIR}
./configure --prefix=${HYPRE_INSTALL_DIR} --enable-bigint --without-superlu --without-openmp
make -j ${NPROCS} && make install
popd

cmake -DHYPRE_DIR=${HYPRE_INSTALL_DIR} ../
make -j ${NPROCS}
