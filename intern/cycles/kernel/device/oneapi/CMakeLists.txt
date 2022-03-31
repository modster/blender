# SPDX-License-Identifier: Apache-2.0
# Copyright 2021-2022 Intel Corporation

cmake_minimum_required(VERSION 3.13)

set(CMAKE_CXX_STANDARD 17)

include(GenerateExportHeader)

project(ONEAPI_KERNEL VERSION 0.1.0 LANGUAGES CXX)
message(STATUS "oneAPI kernel: Used compiler ${CMAKE_CXX_COMPILER}")

set (ONEAPI_KERNEL_SRC
    kernel.h
    kernel.cpp
    image.h
    globals.h
    compat.h
    context_begin.h
    context_end.h
    kernel_templates.h
    device_id.h
    dll_interface_template.h)

add_definitions(
  -DWITH_ONEAPI
  -DONEAPI_EXPORT
)

if(WITH_NANOVDB)
  message(STATUS "NanoVDB has been found and enabled for oneAPI rendering")
  add_definitions(-DWITH_NANOVDB)
  include_directories(
    SYSTEM
    ${NANOVDB_INCLUDE_DIR}
  )
endif()

string(REPLACE "&" ";" CYCLES_INCLUDE_DIRS "${CYCLES_INCLUDE_DIRS_ESCAPED}")

add_library(cycles_kernel_oneapi SHARED ${ONEAPI_KERNEL_SRC})

generate_export_header(cycles_kernel_oneapi)
# Include binary directory in order to find export header
include_directories(${PROJECT_BINARY_DIR})

target_include_directories(cycles_kernel_oneapi PUBLIC ${SYCL_INCLUDE_DIR}/sycl ${SYCL_INCLUDE_DIR} ${LEVEL_ZERO_INCLUDE_DIR})

target_compile_options(cycles_kernel_oneapi PRIVATE -fsycl -fsycl-unnamed-lambda)
if (ORIGINALCMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  # Additionally enable extensions for MSVC project, because oneapi doesn't do it
  target_compile_options(cycles_kernel_oneapi PRIVATE /EHsc)

  # Additionally set proper debug output format for MSVC project, because oneapi doesn't do it
  # Temporaly disable
  # target_compile_options(cycles_kernel_oneapi PRIVATE /Z7)
elseif(UNIX AND NOT APPLE)
  # -fsycl option on compilation and shared library linking is totally necessary (at least on Linux)
  target_link_options(cycles_kernel_oneapi PRIVATE -fsycl)
endif()

if (WITH_CYCLES_ONEAPI_SYCL_HOST_ENABLED)
  add_definitions(-DWITH_ONEAPI_SYCL_HOST_ENABLED)
endif()

#[[
if (UNIX AND NOT APPLE)
  string(TOLOWER ${CMAKE_BUILD_TYPE} LOWERED_CMAKE_BUILD_TYPE)
  if (LOWERED_CMAKE_BUILD_TYPE STREQUAL "debug")
    message(STATUS "oneAPI kernel: Enable debug symbols for oneAPI, because Debug build have been requested")
    # Passing this options (espicially -O0) will enable full debug symbols passing for SYCL execution
    # Which makes a lot of easy debugging on CPU and on GPU (not yet supported) as well
    target_compile_options(cycles_kernel_oneapi PRIVATE -g -O0)
    add_compile_definitions(_DEBUG)
  endif()
endif()
]]#

# oneAPI GPU build parameters
SET (CYCLES_ONEAPI_GPU_COMPILATION_OPTIONS "-internal_options '-ze-opt-large-register-file -ze-opt-regular-grf-kernel integrator_intersect'")
if(UNIX AND NOT APPLE)
  #NOTE(sirgienko) Such direct manupulation of link flags is required, overwise CMake fill fail to set different quotes properly.
  SET(CMAKE_LINK_FLAGS_SYCL "")
  if (WITH_CYCLES_ONEAPI_BINARIES)
    target_compile_options(cycles_kernel_oneapi PRIVATE -fsycl-targets=spir64_gen)
    SET(CMAKE_LINK_FLAGS_SYCL
      "${CMAKE_LINK_FLAGS_SYCL} -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen \"-device ${CYCLES_ONEAPI_AOT_TARGETS} ${CYCLES_ONEAPI_GPU_COMPILATION_OPTIONS}\""
    )
  endif()
  SET(CMAKE_LINK_FLAGS_SYCL
    "${CMAKE_LINK_FLAGS_SYCL} -Xsycl-target-backend=spir64 \"${CYCLES_ONEAPI_GPU_COMPILATION_OPTIONS}\""
  )
  SET_PROPERTY(TARGET cycles_kernel_oneapi APPEND PROPERTY LINK_FLAGS "${CMAKE_LINK_FLAGS_SYCL}")
elseif(ORIGINALCMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  if (WITH_CYCLES_ONEAPI_BINARIES)
    SET (VS2019_CONFIG_FILENAME "vs2019_aot_config_file.props.cmake")
  else()
    SET (VS2019_CONFIG_FILENAME "vs2019_config_file.props.cmake")
  endif()
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}\\${VS2019_CONFIG_FILENAME} ${CMAKE_CURRENT_BINARY_DIR}\\vs2019_config_file.props @ONLY)
  set_target_properties(cycles_kernel_oneapi PROPERTIES VS_USER_PROPS "${CMAKE_CURRENT_BINARY_DIR}\\vs2019_config_file.props")
endif()

# NOTE(sirgienko) Temporary workaround for known problem with compilation (and execution) for GPU with "-g" option.
# So for now, this option will be removed from compilation settings
string(REPLACE "-g" "" tmp_value "${CMAKE_CXX_FLAGS_DEBUG}")
set_property(CACHE CMAKE_CXX_FLAGS_DEBUG PROPERTY VALUE "${tmp_value}")

string(REPLACE "-g" "" tmp_value "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
set_property(CACHE CMAKE_CXX_FLAGS_RELWITHDEBINFO PROPERTY VALUE "${tmp_value}")

get_target_property(ONEAPI_COMPILE_OPTIONS cycles_kernel_oneapi COMPILE_OPTIONS)
list(REMOVE_ITEM ONEAPI_COMPILE_OPTIONS "-g")
set_property(TARGET cycles_kernel_oneapi PROPERTY COMPILE_OPTIONS ${ONEAPI_COMPILE_OPTIONS})

# We need OpenCL from oneAPI installation for oneAPI kernel library
# For searching this certain library, cmake will use library directory, where SYCL_LIBRARY is
FIND_LIBRARY(ONEAPI_OPENCL_LIBRARY
NAMES
  OpenCL
HINTS
  ${SYCL_LIBRARY_DIR}
PATH_SUFFIXES
  lib64 lib
REQUIRED
)
get_filename_component(ONEAPI_OPENCL_LIBRARY_DIR "${ONEAPI_OPENCL_LIBRARY}" DIRECTORY)
link_directories(${ONEAPI_OPENCL_LIBRARY_DIR})


target_link_libraries(cycles_kernel_oneapi PUBLIC ${ONEAPI_OPENCL_LIBRARY})
if (ORIGINALCMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  target_link_libraries(cycles_kernel_oneapi INTERFACE "${SYCL_LIBRARY_DIR}\\sycl$<$<CONFIG:Debug>:d>${CMAKE_LINK_LIBRARY_SUFFIX}")
else()
  target_link_libraries(cycles_kernel_oneapi INTERFACE "${SYCL_LIBRARY_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}sycl${CMAKE_SHARED_LIBRARY_SUFFIX}")
endif()
target_link_libraries(cycles_kernel_oneapi PUBLIC ${LEVEL_ZERO_LIBRARY})

foreach(DIR IN ITEMS ${CYCLES_INCLUDE_DIRS})
  include_directories(${DIR})
endforeach(DIR)
