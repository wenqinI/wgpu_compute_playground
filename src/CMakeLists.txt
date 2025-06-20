set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/wgpu")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/wgpu")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/wgpu")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/common)

add_subdirectory(matmulnbits)

# Create a custom target to copy files
if(WIN32)
  set(SRC)
  set(SRC ${SRC} ${CMAKE_BINARY_DIR}/$<CONFIG>/dxcompiler.dll)
  set(SRC ${SRC} ${CMAKE_BINARY_DIR}/$<CONFIG>/dxil.dll)

  set(DST ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>)

  add_custom_target(
    copy_files ALL
    COMMENT "Copy DLX DLLs"
    COMMAND ${CMAKE_COMMAND} -E make_directory ${DST}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC} ${DST}
    DEPENDS dawn::dawn_proc dawn::dawn_native)
endif()
