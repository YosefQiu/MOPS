#!/bin/bash

module load cudatoolkit/12.4
module load PrgEnv-gnu
module load hip/5.5.1
module load gcc-native/12.3
module load cray-python/3.11.5
module load cmake/3.30.2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-${(%):-%x}}")" && pwd)"
INSTALL_DIR="$(realpath "$SCRIPT_DIR/../third_lib")"
LUA_SCRIPT="$(realpath "$SCRIPT_DIR/download.lua")"
SYCL_SCRIPT="$(realpath "$SCRIPT_DIR/setSYCL.sh")"

lua "$LUA_SCRIPT"

echo "$SYCL_SCRIPT"
source "$SYCL_SCRIPT"

module load cudatoolkit/12.4
module load PrgEnv-gnu
module load hip/5.5.1
module load gcc-native/12.3
module load cray-python/3.11.5
module load cmake/3.30.2

rm -rf build_hip && mkdir -p build_hip
cd build_hip

export HIP_PLATFORM=nvidia

# Use the validated cluster workflow: hipcc as CXX compiler, HIP flags from hipconfig.
cmake .. \
  -DCMAKE_C_COMPILER=gcc \
  -DCMAKE_CXX_COMPILER=$(which hipcc) \
  -DMOPS_BACKEND=HIP \
  -DCMAKE_PREFIX_PATH="$INSTALL_DIR" \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
  -DCMAKE_BUILD_RPATH="$INSTALL_DIR/lib;$INSTALL_DIR/lib64" \
  -DCMAKE_INSTALL_RPATH="$INSTALL_DIR/lib;$INSTALL_DIR/lib64" \
  -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
  -DMOPS_VTK=ON

make -j16
make install

export LD_LIBRARY_PATH=$INSTALL_DIR/lib:$INSTALL_DIR/lib64:$LD_LIBRARY_PATH