set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test)

add_library(catch_main OBJECT impl.cpp)
target_include_directories(catch_main PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp"
)


function(mops_apply_test_deps target)
  target_include_directories(${target} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}   
  )

  target_link_libraries(${target} PRIVATE MOPS)

  target_compile_options(${target} PRIVATE
    -fsycl -Wdeprecated-declarations -fsycl-targets=nvptx64-nvidia-cuda
  )

  target_link_options(${target} PRIVATE
    -pthread -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Wdeprecated-declarations
  )

  vtk_module_autoinit(
    TARGETS ${target}
    MODULES ${VTK_LIBRARIES}
  )
endfunction()


add_executable(all_tests
  $<TARGET_OBJECTS:catch_main>
  ${TEST_SOURCES}
)
mops_apply_test_deps(all_tests)


foreach(src ${TEST_SOURCES})
  get_filename_component(name ${src} NAME_WE)

  add_executable(${name}
    $<TARGET_OBJECTS:catch_main>
    ${src}
  )
  mops_apply_test_deps(${name})
endforeach()
