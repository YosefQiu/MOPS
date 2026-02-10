file(GLOB TUTORIAL_SOURCES *.cpp)

foreach(TUTORIAL_SRC ${TUTORIAL_SOURCES})
    get_filename_component(TUTORIAL_NAME ${TUTORIAL_SRC} NAME_WE)
    
    add_executable(${TUTORIAL_NAME} ${TUTORIAL_SRC})

    target_include_directories(${TUTORIAL_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(${TUTORIAL_NAME} PRIVATE MOPS)

    target_compile_options(${TUTORIAL_NAME} PRIVATE
        -fsycl -Wdeprecated-declarations -fsycl-targets=nvptx64-nvidia-cuda
    )
    
    target_link_options(${TUTORIAL_NAME} PRIVATE
        -pthread -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Wdeprecated-declarations
    )
    
    if (MOPS_VTK)
      vtk_module_autoinit(
        TARGETS ${TUTORIAL_NAME}
        MODULES ${VTK_LIBRARIES}
      )
    endif()
endforeach()
