file(GLOB TEST_SOURCES *.cpp)

foreach(TEST_SRC ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)
    
    add_executable(${TEST_NAME} ${TEST_SRC})
    
    target_include_directories(${TEST_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
    )
    
    target_link_libraries(${TEST_NAME} PRIVATE MOPS)
    
    target_compile_options(${TEST_NAME} PRIVATE
        -fsycl -Wdeprecated-declarations -fsycl-targets=nvptx64-nvidia-cuda
    )
    
    target_link_options(${TEST_NAME} PRIVATE
        -pthread -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Wdeprecated-declarations
    )
    
    vtk_module_autoinit(
        TARGETS ${TEST_NAME}
        MODULES ${VTK_LIBRARIES}
    )
endforeach()