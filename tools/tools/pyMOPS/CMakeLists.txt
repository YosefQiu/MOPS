# 依赖：Python3 + pybind11，要求主 CMake 已通过 -D 指定路径
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

# 构建 Python 模块
add_library(pyMOPS MODULE bindings.cpp)

# 添加头文件路径
target_include_directories(pyMOPS PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
    ../../include            # 你的项目头文件
)

# 链接已有的 core 逻辑库（如 MPASOGrid、MPASOSolution）
target_link_libraries(pyMOPS PRIVATE
    mops_core               # 从主项目中构建的逻辑核心
    ${Python3_LIBRARIES}
)

# OneAPI SYCL 选项（继承主项目）
target_compile_options(pyMOPS PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda)
target_link_options(pyMOPS PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda)

# 设置为 Python 模块，无前缀、.so 后缀
set_target_properties(pyMOPS PROPERTIES PREFIX "" SUFFIX ".so")
set_target_properties(pyMOPS PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/pyMOPS
)