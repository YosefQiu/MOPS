cmake_minimum_required(VERSION 3.12)
project(app)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MOPS REQUIRED)
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

