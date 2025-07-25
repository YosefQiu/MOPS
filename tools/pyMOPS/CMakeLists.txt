find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)


add_library(pyMOPS MODULE bindings.cpp)

target_include_directories(pyMOPS PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)


target_link_libraries(pyMOPS PRIVATE MOPS ${Python3_LIBRARIES})

target_compile_options(pyMOPS PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda)
target_link_options(pyMOPS PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda)

set_target_properties(pyMOPS PROPERTIES PREFIX "" SUFFIX ".so")
set_target_properties(pyMOPS PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/pyMOPS
)