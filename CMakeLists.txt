cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(MOPS VERSION 0.0.1)

set(PRJ_COMPILE_FEATURES)
set(CMAKE_C_COMPILER icx)
set(CMAKE_CXX_COMPILER icpx)

# Define an option for building with VTK support
option(MOPS_USE_VTK "Use VTK" ON)
set(PYTHON_EXPORT ON)

# Find VTK if enabled
if(MOPS_USE_VTK)
    find_package(VTK COMPONENTS
      CommonColor
      CommonCore
      CommonDataModel
      FiltersCore
      FiltersSources
      FiltersGeometry
      IOLegacy
      IOXML
      InteractionStyle
    )

    if(VTK_FOUND)
        message(STATUS "VTK found at: ${VTK_DIR}")
        set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;${VTK_LIBRARY_DIRS}")
    else()
        message(FATAL_ERROR "VTK NOT FOUND!")
    endif()
endif()

find_package(yaml-cpp REQUIRED)
find_package (CUDAToolkit)
find_package(ndarray REQUIRED)
find_package(TBB REQUIRED)
find_package(netCDF)

include_directories(
  include
  ${ndarray_INCLUDE_DIRS}
  ${VTK_INCLUDE_DIRS}
  ${netCDF_INCLUDE_DIR}
)

set(CMAKE_C_FLAGS "-fsycl -fno-fast-math -fsycl-targets=nvptx64-nvidia-cuda")
set(CMAKE_CXX_FLAGS "-fsycl -fno-fast-math -fsycl-targets=nvptx64-nvidia-cuda")
set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)
message(STATUS "CMAKE_C_COMPILER = ${CMAKE_C_COMPILER}")
message(STATUS "CMAKE_CXX_COMPILER = ${CMAKE_CXX_COMPILER}")
message(STATUS "CMAKE_C_FLAGS =  ${CMAKE_C_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS =  ${CMAKE_CXX_FLAGS}")

file(GLOB_RECURSE SOURCE_FILES src/*.c src/*.cpp src/*.cc)
message("Following files are compiled: ${SOURCE_FILES}")

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

set(TARGET_LIBRARIES
    CUDA::cudart
    yaml-cpp::yaml-cpp
    ndarray::ndarray
    netCDF::netcdf
    TBB::tbb
)

if(MOPS_USE_VTK)
    list(APPEND TARGET_LIBRARIES ${VTK_LIBRARIES})
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE ${TARGET_LIBRARIES})

if(MOPS_USE_VTK)
    vtk_module_autoinit(
        TARGETS ${PROJECT_NAME}
        MODULES ${VTK_LIBRARIES}
    )
endif()

target_compile_features(${PROJECT_NAME} PRIVATE ${PRJ_COMPILE_FEATURES})

add_library(mops_core STATIC
    src/MPASOGrid.cpp
    src/MPASOSolution.cpp
    src/MPASOReader.cpp
    src/SYCLKernel.cpp
    src/MOPSApp.cpp
    src/MOPS.cpp
    src/MPASOVisualizer.cpp
    src/MPASOField.cpp
)

set_target_properties(mops_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(mops_core PUBLIC include)
target_link_libraries(mops_core PUBLIC yaml-cpp::yaml-cpp TBB::tbb ${NetCDF_LIBRARIES} ${VTK_LIBRARIES})
vtk_module_autoinit(
    TARGETS mops_core
    MODULES ${VTK_LIBRARIES}
)
if(PYTHON_EXPORT)
    add_subdirectory(tools/pyMOPS)
endif()

install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Targets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/ DESTINATION include)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/MOPSConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    INSTALL_DESTINATION lib/cmake/${PROJECT_NAME}
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    DESTINATION lib/cmake/${PROJECT_NAME}
)

export(EXPORT ${PROJECT_NAME}Targets
    FILE ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake
    NAMESPACE ${PROJECT_NAME}::
)
