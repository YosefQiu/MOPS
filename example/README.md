```
cmake .. -DCMAKE_C_COMPILER=icx \
-DCMAKE_CXX_COMPILER=icpx \
-DnetCDF_DIR=$HOME/lusa/lua_config/third_lib/lib64/cmake/netCDF \
-Dyaml-cpp_DIR=$HOME/lusa/lua_config/third_lib/lib64/cmake/yaml-cpp \
-Dndarray_DIR=$HOME/lusa/lua_config/third_lib/lib/cmake/ndarray \
-Dpybind11_DIR=$HOME/lusa/lua_config/third_lib/share/cmake/pybind11 \
-DVTK_DIR=$HOME/lusa/lua_config/third_lib/lib64/cmake/vtk-9.2 \
-DMOPS_DIR=$HOME/lusa/lua_config/third_lib/lib/cmake/MOPS 
```
