cmake_minimum_required(VERSION 3.12)
project(app)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MOPS REQUIRED)

set(MMPI "/opt/cray/pe/mpich/8.1.25/ofi/cray/10.0")
set(MFAB "/opt/cray/libfabric/1.22.0")
set(MGTL "/opt/cray/pe/mpich/8.1.25/gtl")

include_directories(SYSTEM
    ${MMPI}/include
  )
link_directories(
    ${MMPI}/lib
    ${MFAB}/lib64
    ${MGTL}/lib
  )

set(CMAKE_BUILD_RPATH "${MMPI}/lib;${MFAB}/lib64")
set(CUDA_LIB_PATH "/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/cuda/11.7/lib64")

link_directories(
    ${CUDA_LIB_PATH})

add_executable(app main.cpp)

target_link_libraries(app PRIVATE MOPS::MOPS)

target_include_directories(app PRIVATE 
    /global/common/software/nersc9/intel/oneapi/compiler/2024.1/include/
    /global/common/software/nersc9/intel/oneapi/compiler/2024.1/include/sycl/
)

target_link_libraries(app PRIVATE
    sycl
    OpenCL
)

target_link_libraries(app PRIVATE mpi mpi_gtl_cuda cudart fabric)
