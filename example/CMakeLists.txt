cmake_minimum_required(VERSION 3.12)
project(app)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MOPS REQUIRED)
add_executable(app main.cpp)

target_link_libraries(app PRIVATE MOPS::MOPS)

target_include_directories(app PRIVATE 
    /global/common/software/nersc9/intel/oneapi/compiler/2024.1/include/
)

target_compile_options(${PROJECT_NAME} PRIVATE 
  -pthread -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Wdeprecated-declarations -fp-model=precise
)

target_link_options(${PROJECT_NAME} PRIVATE
  -pthread -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Wdeprecated-declarations -fp-model=precise
)