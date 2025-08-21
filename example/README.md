# MOPS Example Build Instructions

This project supports two build modes:

1. **Using Intel OneAPI Compiler (icx/icpx)**
2. **Using GNU Compiler (g++) to link prebuilt libMOPS.so**

---

## Prerequisites

- MOPS has been built and installed to `third_lib/`:

  ```
  third_lib/
    ├── include/                # MOPS headers
    ├── lib/libMOPS.so          # Precompiled MOPS shared library
    └── lib/cmake/MOPS/         # MOPSConfig.cmake
  ```

- Intel OneAPI compiler (2024.1 or later) is installed and sourced.

---

## Build with Intel OneAPI Compiler (Full SYCL Build)

Use this if you want to compile **both MOPS and your application with icpx**.

```bash
cmake ..   -DCMAKE_C_COMPILER=icx   -DCMAKE_CXX_COMPILER=icpx   -DCMAKE_PREFIX_PATH=$PWD/../../third_lib   -DCMAKE_BUILD_TYPE=Release
make -j
```

This will compile your entire project (including SYCL code) using Intel OneAPI DPC++ compiler.

---

## Build with GNU Compiler (Link Prebuilt libMOPS.so)

Use this if you already have a precompiled `libMOPS.so` (built with icpx) and just want to compile your **main.cpp with g++**.

```bash
cmake ..   -DCMAKE_CXX_COMPILER=g++   -DCMAKE_PREFIX_PATH=$PWD/../../third_lib   -DCMAKE_BUILD_TYPE=Release
make -j
```

### Important:

- Your `main.cpp` should **only call MOPS API functions**, and must **NOT contain SYCL kernel code**.

- Ensure the following include paths are added in your CMakeLists.txt:

  ```cmake
  target_include_directories(app PRIVATE
      /path/to/oneapi/compiler/latest/include/
      /path/to/oneapi/compiler/latest/include/sycl/
  )
  ```

- Ensure runtime libraries are found at runtime:

  ```bash
  export LD_LIBRARY_PATH=/path/to/third_lib/lib:/path/to/oneapi/compiler/latest/lib:$LD_LIBRARY_PATH
  ```

---

## Example Directory Structure:

```
project/
  ├── CMakeLists.txt
  ├── main.cpp
  ├── build/
  └── third_lib/
        ├── include/
        ├── lib/libMOPS.so
        └── lib/cmake/MOPS/MOPSConfig.cmake
```

---

## Notes:

- Intel OneAPI SYCL headers (sycl.hpp) use `<CL/__spirv/spirv_types.hpp>`, so when using g++, you must manually add `include/sycl/` as an include path.
- Ensure your `MOPSConfig.cmake` properly exports MOPS::MOPS target linking to SYCL/OpenCL runtime.


