cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(MOPS VERSION 0.1.0)

option(MOPS_TESTING       "Enable test builds"                    ON)
option(MOPS_PYTHON        "Enable Python bindings (tools/pyMOPS)" ON)
option(MOPS_TUTORIAL      "Enable MOPS Tutorial examples"         ON)
option(MOPS_VTK           "Enable VTK-based visualization and IO" ON) 
option(MOPS_MPI           "Enable MPI support"                    OFF)
option(MOPS_CLI           "Enable CLI support"                    ON)
option(MOPS_DEBUG         "Enable debug logging"                  ON)
option(MOPS_TIMING        "Enable internal timing profiler"       ON)

# GPU Backend Selection: SYCL (default), CUDA, HIP, or CPU
set(MOPS_BACKEND "SYCL" CACHE STRING "GPU backend: SYCL, CUDA, HIP, or CPU")
set_property(CACHE MOPS_BACKEND PROPERTY STRINGS "SYCL" "CUDA" "HIP" "CPU")

set(BUILD_SHARED_LIBS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "") 


if (MOPS_DEBUG)
  add_compile_definitions(_DEBUG=1)
else()
  add_compile_definitions(_DEBUG=0)
endif()

if (MOPS_TIMING)
  add_compile_definitions(MOPS_ENABLE_TIMING=1)
endif()

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
else()
  add_compile_definitions(MOPS_MPI=0)
endif()



find_package(yaml-cpp REQUIRED)
find_package(netCDF REQUIRED)
find_package(ndarray REQUIRED)
find_package(TBB REQUIRED)

# Get Git revision if available
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${CMAKE_SOURCE_DIR}/.git")
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE SHORT_SHA
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(SHORT_SHA)
        set(REVISION ${SHORT_SHA} CACHE STRING "git short sha" FORCE)
    else()
        set(REVISION "unknown")
    endif()
else()
    message(STATUS "Git not found or not a git repository, using default version")
    set(REVISION "unknown")
endif()

set(PROJECT_VERSION_ ${REVISION})

# generate version.h
include_directories(${CMAKE_BINARY_DIR})
configure_file("${CMAKE_SOURCE_DIR}/src/version.h.in" "${CMAKE_BINARY_DIR}/version.h" @ONLY)

#=============================================================================
# GPU Backend Configuration
#=============================================================================
if(MOPS_BACKEND STREQUAL "SYCL")
  message(STATUS "Using SYCL backend")
  add_compile_definitions(MOPS_USE_CPU=0)
  add_compile_definitions(MOPS_USE_GPU=1)
  add_compile_definitions(MOPS_USE_SYCL=1)
  add_compile_definitions(MOPS_USE_CUDA=0)
  add_compile_definitions(MOPS_USE_HIP=0)
  add_compile_definitions(MOPS_USE_TBB=0)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -Wdeprecated-declarations -fsycl-targets=nvptx64-nvidia-cuda")

elseif(MOPS_BACKEND STREQUAL "CUDA")
  message(STATUS "Using CUDA backend")
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wdeprecated-declarations")
  add_compile_definitions(MOPS_USE_CPU=0)
  add_compile_definitions(MOPS_USE_GPU=1)
  add_compile_definitions(MOPS_USE_SYCL=0)
  add_compile_definitions(MOPS_USE_CUDA=1)
  add_compile_definitions(MOPS_USE_HIP=0)
  add_compile_definitions(MOPS_USE_TBB=0)

elseif(MOPS_BACKEND STREQUAL "HIP")
  message(STATUS "Using HIP backend")
  find_package(CUDAToolkit REQUIRED)
  add_compile_definitions(MOPS_USE_CPU=0)
  add_compile_definitions(MOPS_USE_GPU=1)
  add_compile_definitions(MOPS_USE_SYCL=0)
  add_compile_definitions(MOPS_USE_CUDA=0)
  add_compile_definitions(MOPS_USE_HIP=1)
  add_compile_definitions(MOPS_USE_TBB=0)

  # Keep hipcc for compilation, but force host g++ for final linking.
  # This avoids nvcc/hipcc link-stage failures on absolute third-party .so paths.
  set(CMAKE_CXX_CREATE_SHARED_LIBRARY
      "g++ <CMAKE_SHARED_LIBRARY_CXX_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS> <SONAME_FLAG><TARGET_SONAME> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")
    set(CMAKE_CXX_CREATE_SHARED_MODULE
      "g++ <CMAKE_SHARED_MODULE_CXX_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_MODULE_CREATE_CXX_FLAGS> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")
  set(CMAKE_CXX_LINK_EXECUTABLE
      "g++ <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")

elseif(MOPS_BACKEND STREQUAL "CPU")
  message(STATUS "Using CPU backend (no GPU acceleration)")
  add_compile_definitions(MOPS_USE_CPU=1)
  add_compile_definitions(MOPS_USE_GPU=0)
  add_compile_definitions(MOPS_USE_SYCL=0)
  add_compile_definitions(MOPS_USE_CUDA=0)
  add_compile_definitions(MOPS_USE_HIP=0)
  add_compile_definitions(MOPS_USE_TBB=1)
  
else()
  message(FATAL_ERROR "Invalid MOPS_BACKEND: ${MOPS_BACKEND}. Use SYCL, CUDA, HIP, or CPU.")
endif()

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

# Backend-selective source collection: always include core/common, then append selected backend only.
set(SOURCE_SRC)

file(GLOB_RECURSE SOURCE_SRC_CORE
  ${CMAKE_SOURCE_DIR}/src/Common/*.c
  ${CMAKE_SOURCE_DIR}/src/Common/*.cpp
  ${CMAKE_SOURCE_DIR}/src/Common/*.cc
  ${CMAKE_SOURCE_DIR}/src/Common/*.cxx
  ${CMAKE_SOURCE_DIR}/src/Core/*.c
  ${CMAKE_SOURCE_DIR}/src/Core/*.cpp
  ${CMAKE_SOURCE_DIR}/src/Core/*.cc
  ${CMAKE_SOURCE_DIR}/src/Core/*.cxx
  ${CMAKE_SOURCE_DIR}/src/IO/*.c
  ${CMAKE_SOURCE_DIR}/src/IO/*.cpp
  ${CMAKE_SOURCE_DIR}/src/IO/*.cc
  ${CMAKE_SOURCE_DIR}/src/IO/*.cxx
  ${CMAKE_SOURCE_DIR}/src/Utils/*.c
  ${CMAKE_SOURCE_DIR}/src/Utils/*.cpp
  ${CMAKE_SOURCE_DIR}/src/Utils/*.cc
  ${CMAKE_SOURCE_DIR}/src/Utils/*.cxx
)
list(FILTER SOURCE_SRC_CORE EXCLUDE REGEX "/src/Utils/CUDACommon/.*\\.(c|cc|cpp|cxx)$")
list(FILTER SOURCE_SRC_CORE EXCLUDE REGEX "/src/Utils/Common/.*\\.(c|cc|cpp|cxx)$")
list(APPEND SOURCE_SRC ${SOURCE_SRC_CORE})

file(GLOB_RECURSE SOURCE_SRC_GPU_COMMON
  ${CMAKE_SOURCE_DIR}/src/GPU/Common/*.c
  ${CMAKE_SOURCE_DIR}/src/GPU/Common/*.cpp
  ${CMAKE_SOURCE_DIR}/src/GPU/Common/*.cc
  ${CMAKE_SOURCE_DIR}/src/GPU/Common/*.cxx
  ${CMAKE_SOURCE_DIR}/src/CPU/Common/*.c
  ${CMAKE_SOURCE_DIR}/src/CPU/Common/*.cpp
  ${CMAKE_SOURCE_DIR}/src/CPU/Common/*.cc
  ${CMAKE_SOURCE_DIR}/src/CPU/Common/*.cxx
)
list(APPEND SOURCE_SRC ${SOURCE_SRC_GPU_COMMON})

if(MOPS_BACKEND STREQUAL "SYCL")
  file(GLOB_RECURSE SOURCE_SRC_GPU_BACKEND
    ${CMAKE_SOURCE_DIR}/src/GPU/SYCL/*.c
    ${CMAKE_SOURCE_DIR}/src/GPU/SYCL/*.cpp
    ${CMAKE_SOURCE_DIR}/src/GPU/SYCL/*.cc
    ${CMAKE_SOURCE_DIR}/src/GPU/SYCL/*.cxx
  )
elseif(MOPS_BACKEND STREQUAL "CUDA")
  file(GLOB_RECURSE SOURCE_SRC_GPU_BACKEND
    ${CMAKE_SOURCE_DIR}/src/GPU/CUDA/*.c
    ${CMAKE_SOURCE_DIR}/src/GPU/CUDA/*.cpp
    ${CMAKE_SOURCE_DIR}/src/GPU/CUDA/*.cc
    ${CMAKE_SOURCE_DIR}/src/GPU/CUDA/*.cxx
    ${CMAKE_SOURCE_DIR}/src/GPU/CUDA/*.cu
  )
elseif(MOPS_BACKEND STREQUAL "HIP")
  file(GLOB_RECURSE SOURCE_SRC_GPU_BACKEND
    ${CMAKE_SOURCE_DIR}/src/GPU/HIP/*.c
    ${CMAKE_SOURCE_DIR}/src/GPU/HIP/*.cpp
    ${CMAKE_SOURCE_DIR}/src/GPU/HIP/*.cc
    ${CMAKE_SOURCE_DIR}/src/GPU/HIP/*.cxx
    ${CMAKE_SOURCE_DIR}/src/GPU/HIP/*.cu
    ${CMAKE_SOURCE_DIR}/src/GPU/HIP/*.hip
  )
else()
  file(GLOB_RECURSE SOURCE_SRC_GPU_BACKEND
    ${CMAKE_SOURCE_DIR}/src/CPU/TBB/*.c
    ${CMAKE_SOURCE_DIR}/src/CPU/TBB/*.cpp
    ${CMAKE_SOURCE_DIR}/src/CPU/TBB/*.cc
    ${CMAKE_SOURCE_DIR}/src/CPU/TBB/*.cxx
  )
endif()
list(APPEND SOURCE_SRC ${SOURCE_SRC_GPU_BACKEND})

list(LENGTH SOURCE_SRC MOPS_SELECTED_SOURCE_COUNT)
message(STATUS "Selected source files : ${MOPS_SELECTED_SOURCE_COUNT}")

add_library(${PROJECT_NAME} SHARED ${SOURCE_SRC})

if(MOPS_BACKEND STREQUAL "HIP")
  execute_process(
    COMMAND hipconfig --cxx_config
    OUTPUT_VARIABLE HIP_CXX_CONFIG_RAW
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  separate_arguments(HIP_CXX_CONFIG NATIVE_COMMAND "${HIP_CXX_CONFIG_RAW}")

  set(HIP_DEVICE_SOURCES ${SOURCE_SRC_GPU_BACKEND})
  list(FILTER HIP_DEVICE_SOURCES INCLUDE REGEX ".*\\.(hip|cu)$")
  # HIPKernel.cu is host-side helper code; keep it out of device compilation mode.
  list(REMOVE_ITEM HIP_DEVICE_SOURCES "${CMAKE_SOURCE_DIR}/src/GPU/HIP/Kernel/HIPKernel.cu")
  foreach(_hip_src IN LISTS HIP_DEVICE_SOURCES)
    set_source_files_properties(${_hip_src} PROPERTIES LANGUAGE CXX)
    set_source_files_properties(${_hip_src} PROPERTIES COMPILE_DEFINITIONS "MOPS_HIP_DEVICE_BUILD=1")
    set_source_files_properties(${_hip_src} PROPERTIES COMPILE_OPTIONS "-x;cu;-Xcompiler;-fPIC;-fPIC;${HIP_CXX_CONFIG}")
  endforeach()

  set_source_files_properties(
    ${CMAKE_SOURCE_DIR}/src/GPU/HIP/Kernel/HIPKernel.cu
    PROPERTIES
      LANGUAGE CXX
      COMPILE_OPTIONS "-x;c++"
  )
endif()

target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
)

target_link_libraries(${PROJECT_NAME} PUBLIC
  ${VTK_LIBRARIES}
  yaml-cpp::yaml-cpp
  TBB::tbb
  netCDF::netcdf
  ndarray::ndarray
)

if(MOPS_BACKEND STREQUAL "HIP")
  target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::cudart)
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  EXPORT_NAME ${PROJECT_NAME}
)

if(MOPS_BACKEND STREQUAL "CUDA")
  set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
endif()

if (MOPS_VTK)
  vtk_module_autoinit(TARGETS ${PROJECT_NAME} MODULES ${VTK_LIBRARIES})
endif()

if (MOPS_CLI)
  set(CLI_NAME "${PROJECT_NAME}_CLI")
  add_executable(${CLI_NAME} CLI/main.cpp)
  if (MOPS_MPI)
    target_link_libraries(${CLI_NAME} PRIVATE mpi fabric)
  endif()
  target_link_libraries(${CLI_NAME} PRIVATE ${PROJECT_NAME})

  if(MOPS_BACKEND STREQUAL "SYCL")
    target_compile_options(${CLI_NAME} PRIVATE
      -pthread -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Wdeprecated-declarations -fp-model=precise
    )

    target_link_options(${CLI_NAME} PRIVATE
      -pthread -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Wdeprecated-declarations -fp-model=precise
    )
  endif()

  if (MOPS_VTK)
    vtk_module_autoinit(TARGETS ${CLI_NAME} MODULES ${VTK_LIBRARIES})
  endif()
endif()


if(MOPS_PYTHON)
  find_package(pybind11 REQUIRED)
  add_subdirectory(tools/pyMOPS)
endif()

if(MOPS_TESTING)
  if(MOPS_BACKEND STREQUAL "SYCL")
    add_subdirectory(test)
  else()
    message(STATUS "Skipping test/: current backend is ${MOPS_BACKEND}, but test suite is SYCL-only.")
  endif()
endif()

if (MOPS_TUTORIAL)
  add_subdirectory(tutorial)
endif()

add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

message(STATUS "========== MOPS_Tutorial Configuration Summary ==========")
message(STATUS "Project Name          : ${PROJECT_NAME}")
message(STATUS "Project Version       : ${PROJECT_VERSION}")
message(STATUS "Compiler              : ${CMAKE_CXX_COMPILER}")
message(STATUS "GPU Backend           : ${MOPS_BACKEND}")
message(STATUS "CXX Flags             : ${CMAKE_CXX_FLAGS}")
message(STATUS "---------------------------------------------------------")

# Option status
message(STATUS "ENABLE_VTK            : ${MOPS_VTK}")
message(STATUS "ENABLE_PYTHON         : ${MOPS_PYTHON}")
message(STATUS "ENABLE_TESTING        : ${MOPS_TESTING}")
message(STATUS "ENABLE_TUTORIAL       : ${MOPS_TUTORIAL}")
message(STATUS "ENABLE_MPI            : ${MOPS_MPI}")
message(STATUS "ENABLE_DEBUG          : ${MOPS_DEBUG}")

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
    ${CMAKE_SOURCE_DIR}/src/GPU/SYCL/Kernel
    ${CMAKE_SOURCE_DIR}/src/GPU/CUDA
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
