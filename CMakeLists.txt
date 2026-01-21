cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(MOPS VERSION 0.1.0)

option(MOPS_TESTING       "Enable test builds"                    OFF)
option(MOPS_PYTHON        "Enable Python bindings (tools/pyMOPS)" ON)
option(MOPS_TUTORIAL      "Enable MOPS Tutorial examples"         ON)
option(MOPS_VTK           "Enable VTK-based visualization and IO" ON) 
option(MOPS_MPI           "Enable MPI support"                    OFF)
option(MOPS_CLI           "Enable CLI support"                    ON)

set(BUILD_SHARED_LIBS ON)
set(CMAKE_C_COMPILER icpx)
set(CMAKE_CXX_COMPILER icpx)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "") 


if (MOPS_VTK)
  add_compile_definitions(MOPS_VTK=1)
  find_package(VTK REQUIRED)
  set(VTK_LIBRARIES
    VTK::CommonColor
    VTK::CommonCore
    VTK::CommonDataModel
    VTK::FiltersCore
    VTK::FiltersSources
    VTK::FiltersGeometry
    VTK::IOLegacy
    VTK::IOXML
    VTK::InteractionStyle
  )
else()
  add_compile_definitions(MOPS_VTK=0)
  set(VTK_LIBRARIES "") 
endif()

if (MOPS_MPI)
  set(MMPI "/opt/cray/pe/mpich/8.1.25/ofi/cray/10.0")
  set(MFAB "/opt/cray/libfabric/1.22.0")
  set(MGTL "/opt/cray/pe/mpich/8.1.25/gtl")
  add_compile_definitions(MOPS_MPI=1)

  include_directories(SYSTEM
    ${MMPI}/include
  )
  link_directories(
    ${MMPI}/lib
    ${MFAB}/lib64
    ${MGTL}/lib
  )
  set(CMAKE_BUILD_RPATH "${MMPI}/lib;${MFAB}/lib64")
  set(CUDA_LIB_PATH "/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/cuda/11.7/targets/x86_64-linux/lib/")

  link_directories(${CUDA_LIB_PATH})
  include_directories("${CUDA_LIB_PATH}/../include") 

  # RPATH at build and runtime
  set(CMAKE_BUILD_RPATH "${CMAKE_BUILD_RPATH};${CUDA_LIB_PATH}")
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH};${CUDA_LIB_PATH}")
else()
  add_compile_definitions(MOPS_MPI=0)
endif()



find_package(yaml-cpp REQUIRED)
find_package(netCDF REQUIRED)
find_package(ndarray REQUIRED)
find_package(TBB REQUIRED)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}-fsycl -Wdeprecated-declarations -fsycl-targets=nvptx64-nvidia-cuda")
set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)
message(STATUS "CMAKE_C_COMPILER = ${CMAKE_C_COMPILER}")
message(STATUS "CMAKE_CXX_COMPILER = ${CMAKE_CXX_COMPILER}")
message(STATUS "CMAKE_CXX_FLAGS =  ${CMAKE_CXX_FLAGS}")


file(GLOB_RECURSE SOURCE_HDR
  ${CMAKE_SOURCE_DIR}/include/*.h
  ${CMAKE_SOURCE_DIR}/include/*.hpp
  ${CMAKE_SOURCE_DIR}/src/*.h
  ${CMAKE_SOURCE_DIR}/src/*.hpp
)

file(GLOB_RECURSE SOURCE_SRC
  ${CMAKE_SOURCE_DIR}/src/*.c
  ${CMAKE_SOURCE_DIR}/src/*.cpp
  ${CMAKE_SOURCE_DIR}/src/*.cc
  ${CMAKE_SOURCE_DIR}/src/*.cxx
)

add_library(${PROJECT_NAME} SHARED ${SOURCE_SRC})
target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
)

# Ensure SYCL offloading flags are applied to the core library as well
target_compile_options(${PROJECT_NAME} PRIVATE
  -fsycl
  -fsycl-targets=nvptx64-nvidia-cuda
  -Wdeprecated-declarations
)
target_link_options(${PROJECT_NAME} PRIVATE
  -fsycl
  -fsycl-targets=nvptx64-nvidia-cuda
  -Wdeprecated-declarations
)

target_link_libraries(${PROJECT_NAME} PUBLIC
  ${VTK_LIBRARIES}
  yaml-cpp::yaml-cpp
  TBB::tbb
  netCDF::netcdf
  ndarray::ndarray
)

set_target_properties(${PROJECT_NAME} PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  EXPORT_NAME ${PROJECT_NAME}
)

if (MOPS_VTK)
  vtk_module_autoinit(TARGETS ${PROJECT_NAME} MODULES ${VTK_LIBRARIES})
endif()

if (MOPS_CLI)
  set(CLI_NAME "${PROJECT_NAME}_CLI")
  add_executable(${CLI_NAME} CLI/main.cpp)
  if (MOPS_MPI)
    target_link_libraries(${CLI_NAME} PRIVATE mpi mpi_gtl_cuda fabric cudart)
  endif()
  target_link_libraries(${CLI_NAME} PRIVATE ${PROJECT_NAME})

  target_compile_options(${CLI_NAME} PRIVATE 
    -pthread -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Wdeprecated-declarations -fp-model=precise
  )

  target_link_options(${CLI_NAME} PRIVATE
    -pthread -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Wdeprecated-declarations -fp-model=precise
  )

  if (MOPS_VTK)
    vtk_module_autoinit(TARGETS ${CLI_NAME} MODULES ${VTK_LIBRARIES})
  endif()
endif()


if(MOPS_PYTHON)
  find_package(pybind11 REQUIRED)
  add_subdirectory(tools/pyMOPS)
endif()

if(MOPS_TESTING)
  add_subdirectory(test)
endif()

if (MOPS_TUTORIAL)
  add_subdirectory(tutorial)
endif()

add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

message(STATUS "========== MOPS Configuration Summary ==========")
message(STATUS "Project Name          : ${PROJECT_NAME}")
message(STATUS "Project Version       : ${PROJECT_VERSION}")
message(STATUS "Compiler              : ${CMAKE_CXX_COMPILER}")
message(STATUS "SYCL Flags            : ${CMAKE_CXX_FLAGS}")
message(STATUS "---------------------------------------------------------")

# Option status
message(STATUS "ENABLE_VTK            : ${MOPS_VTK}")
message(STATUS "ENABLE_PYTHON         : ${MOPS_PYTHON}")
message(STATUS "ENABLE_TESTING        : ${MOPS_TESTING}")
message(STATUS "ENABLE_TUTORIAL       : ${MOPS_TUTORIAL}")
message(STATUS "ENABLE_MPI            : ${MOPS_MPI}")

# Dependency status
if(MOPS_VTK)
  message(STATUS "VTK_FOUND             : ${VTK_FOUND}")
  message(STATUS "VTK Modules Linked    : ${VTK_LIBRARIES}")
endif()

if (MOPS_MPI)
  message(STATUS "MPI_FOUND             : ${MPI_FOUND}")
  message(STATUS "MPI Include Path      : ${MMPI}/include")
  message(STATUS "MPI Libraries         : ${MMPI}/lib/libmpi.so, ${MFAB}/lib64/libfabric.so")
endif()

message(STATUS "yaml-cpp_FOUND        : ${yaml-cpp_FOUND}")
message(STATUS "TBB_FOUND             : ${TBB_FOUND}")
message(STATUS "NetCDF_FOUND          : ${netCDF_FOUND}")
message(STATUS "ndarray_FOUND         : ${ndarray_FOUND}")

if (MOPS_PYTHON)
  message(STATUS "pybind11_FOUND        : ${pybind11_FOUND}")
  message(STATUS "pybind11_INCLUDE_DIRS : ${pybind11_INCLUDE_DIRS}")
endif()

message(STATUS "=========================================================")

install(
  DIRECTORY ${CMAKE_SOURCE_DIR}/include/
  DESTINATION include
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

install(
  DIRECTORY
    ${CMAKE_SOURCE_DIR}/src/Core
    ${CMAKE_SOURCE_DIR}/src/IO
    ${CMAKE_SOURCE_DIR}/src/SYCL
    ${CMAKE_SOURCE_DIR}/src/Utils
  DESTINATION include
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}Targets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

install(
  EXPORT ${PROJECT_NAME}Targets
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION lib/cmake/${PROJECT_NAME}
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
  "${CMAKE_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION lib/cmake/${PROJECT_NAME}
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
  DESTINATION lib/cmake/${PROJECT_NAME}
)
