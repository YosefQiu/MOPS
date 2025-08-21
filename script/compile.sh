#!/bin/bash

module load gcc-native/12.3
module load cray-hdf5/1.12.2.9
module load cray-python/3.11.5


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="${SCRIPT_DIR}/script"
INSTALL_DIR="$(realpath "$SCRIPT_DIR/../third_lib")"
LUA_SCRIPT="$(realpath "$SCRIPT_DIR/download.lua")" 
SYCL_SCRIPT="$(realpath "$SCRIPT_DIR/setSYCL.sh")"

lua "$LUA_SCRIPT"

echo $SYCL_SCRIPT
source $SYCL_SCRIPT

mkdir -p build
cd build

cmake .. \
  -DCMAKE_C_COMPILER=icx \
  -DCMAKE_CXX_COMPILER=icpx \
  -DnetCDF_DIR=$INSTALL_DIR/lib64/cmake/netCDF \
  -Dyaml-cpp_DIR=$INSTALL_DIR/lib64/cmake/yaml-cpp \
  -Dndarray_DIR=$INSTALL_DIR/lib/cmake/ndarray \
  -Dpybind11_DIR=$INSTALL_DIR/share/cmake/pybind11 \
  -DVTK_DIR=$INSTALL_DIR/lib64/cmake/vtk-9.2 \
  -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR

make -j16
make install
