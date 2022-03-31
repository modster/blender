# SPDX-License-Identifier: Apache-2.0
# Copyright 2011-2022 Blender Foundation

set(INC
  ..
  ../../glew-mx
)

set(INC_SYS
  ${GLEW_INCLUDE_DIR}
  ../../../extern/clew/include
)

if(WITH_CYCLES_DEVICE_OPTIX OR WITH_CYCLES_DEVICE_CUDA)
  if(WITH_CUDA_DYNLOAD)
    list(APPEND INC
      ../../../extern/cuew/include
    )
    add_definitions(-DWITH_CUDA_DYNLOAD)
  else()
    list(APPEND INC_SYS
      ${CUDA_TOOLKIT_INCLUDE}
    )
    add_definitions(-DCYCLES_CUDA_NVCC_EXECUTABLE="${CUDA_NVCC_EXECUTABLE}")
  endif()
endif()

if(WITH_CYCLES_DEVICE_HIP AND WITH_HIP_DYNLOAD)
  list(APPEND INC
    ../../../extern/hipew/include
  )
  add_definitions(-DWITH_HIP_DYNLOAD)
endif()

set(SRC_BASE
  device.cpp
  denoise.cpp
  graphics_interop.cpp
  kernel.cpp
  memory.cpp
  queue.cpp
)

set(SRC_CPU
  cpu/device.cpp
  cpu/device.h
  cpu/device_impl.cpp
  cpu/device_impl.h
  cpu/kernel.cpp
  cpu/kernel.h
  cpu/kernel_function.h
  cpu/kernel_thread_globals.cpp
  cpu/kernel_thread_globals.h
)

set(SRC_CUDA
  cuda/device.cpp
  cuda/device.h
  cuda/device_impl.cpp
  cuda/device_impl.h
  cuda/graphics_interop.cpp
  cuda/graphics_interop.h
  cuda/kernel.cpp
  cuda/kernel.h
  cuda/queue.cpp
  cuda/queue.h
  cuda/util.cpp
  cuda/util.h
)

set(SRC_HIP
  hip/device.cpp
  hip/device.h
  hip/device_impl.cpp
  hip/device_impl.h
  hip/graphics_interop.cpp
  hip/graphics_interop.h
  hip/kernel.cpp
  hip/kernel.h
  hip/queue.cpp
  hip/queue.h
  hip/util.cpp
  hip/util.h
)

set(SRC_ONEAPI
  oneapi/sycl.h
  oneapi/device_impl.cpp
  oneapi/device_impl.h
  oneapi/device.cpp
  oneapi/device.h
  oneapi/queue.cpp
  oneapi/queue.h
  oneapi/util.cpp
  oneapi/util.h
)

set(SRC_DUMMY
  dummy/device.cpp
  dummy/device.h
)

set(SRC_MULTI
  multi/device.cpp
  multi/device.h
)

set(SRC_METAL
  metal/bvh.mm
  metal/bvh.h
  metal/device.mm
  metal/device.h
  metal/device_impl.mm
  metal/device_impl.h
  metal/kernel.mm
  metal/kernel.h
  metal/queue.mm
  metal/queue.h
  metal/util.mm
  metal/util.h
)

set(SRC_OPTIX
  optix/device.cpp
  optix/device.h
  optix/device_impl.cpp
  optix/device_impl.h
  optix/queue.cpp
  optix/queue.h
  optix/util.h
)

set(SRC_HEADERS
  device.h
  denoise.h
  graphics_interop.h
  memory.h
  kernel.h
  queue.h
)

set(SRC
  ${SRC_BASE}
  ${SRC_CPU}
  ${SRC_CUDA}
  ${SRC_HIP}
  ${SRC_DUMMY}
  ${SRC_MULTI}
  ${SRC_OPTIX}
  ${SRC_ONEAPI}
  ${SRC_HEADERS}
)

set(LIB
  cycles_kernel
  cycles_util
  ${CYCLES_GL_LIBRARIES}
)

if(WITH_CYCLES_DEVICE_OPTIX OR WITH_CYCLES_DEVICE_CUDA)
  if(WITH_CUDA_DYNLOAD)
    list(APPEND LIB
      extern_cuew
    )
  else()
    list(APPEND LIB
      ${CUDA_CUDA_LIBRARY}
    )
  endif()
endif()

if(WITH_CYCLES_DEVICE_HIP AND WITH_HIP_DYNLOAD)
  list(APPEND LIB
    extern_hipew
  )
endif()

# Build kernel here for linking with device below
if (WITH_CYCLES_DEVICE_ONEAPI)
  include(ExternalProject)

  add_definitions(-DWITH_ONEAPI)
  if (WITH_CYCLES_ONEAPI_SYCL_HOST_ENABLED)
    add_definitions(-DWITH_ONEAPI_SYCL_HOST_ENABLED)
  endif()
  set(oneapi_kernel_library_filename ${CMAKE_SHARED_LIBRARY_PREFIX}cycles_kernel_oneapi${CMAKE_SHARED_LIBRARY_SUFFIX})

  # Pass a lot of include directories
  get_property(CORRECT_CYCLES_INCLUDE_DIRS DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
  string(REPLACE ";" "&" CORRECT_CYCLES_INCLUDE_DIRS_ESCAPED "${CORRECT_CYCLES_INCLUDE_DIRS}")

  # Get sycl library directory
  get_filename_component(SYCL_LIBRARY_DIR "${SYCL_LIBRARY}" DIRECTORY)

  set (ONEAPI_KERNEL_BUILD_TYPE ${CMAKE_BUILD_TYPE})
  if (WITH_CYCLES_ONEAPI_BINARIES AND NOT GENERATOR_IS_MULTI_CONFIG)
    #NOTE(sirginko) Disable this, when GPU AOT will work with Debug information (for Debug and RelWithDebInfo builds)
    message(STATUS "Change ONEAPI_KERNEL_BUILD_TYPE value from \"${ONEAPI_KERNEL_BUILD_TYPE}\" to \"Release\", because other builds don't work for GPU AOT yet")
    set(ONEAPI_KERNEL_BUILD_TYPE "Release")
  endif()

  if (GENERATOR_IS_MULTI_CONFIG AND CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(ONEAPI_KERNEL_CMAKE_GENERATOR_TOOLSET "Intel(R) oneAPI DPC++ Compiler 2022")
    ExternalProject_Add(
      oneapi_kernel
      DOWNLOAD_COMMAND ""
      SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../kernel/device/oneapi
      CMAKE_ARGS
      -DCMAKE_CXX_COMPILER:STRING=${SYCL_DPCPP_COMPILER}
      -DCMAKE_GENERATOR_TOOLSET=${ONEAPI_KERNEL_CMAKE_GENERATOR_TOOLSET}
      -DWITH_CYCLES_ONEAPI_BINARIES:BOOL=${WITH_CYCLES_ONEAPI_BINARIES}
      -DCYCLES_ONEAPI_SYCL_TARGET:STRING=${CYCLES_ONEAPI_SYCL_TARGET}
      -DCYCLES_ONEAPI_AOT_TARGETS:STRING=${CYCLES_ONEAPI_AOT_TARGETS}
      -DCYCLES_ONEAPI_GPU_COMPILATION_OPTIONS:STRING=${CYCLES_ONEAPI_GPU_COMPILATION_OPTIONS}
      -DWITH_CYCLES_ONEAPI_SYCL_HOST_ENABLED:BOOL=${WITH_CYCLES_ONEAPI_SYCL_HOST_ENABLED}
      -DWITH_NANOVDB:BOOL=${WITH_NANOVDB}
      -DNANOVDB_INCLUDE_DIR=${NANOVDB_INCLUDE_DIR}
      -DSYCL_INCLUDE_DIR=${SYCL_INCLUDE_DIR}
      -DSYCL_LIBRARY_DIR=${SYCL_LIBRARY_DIR}
      -DLEVEL_ZERO_LIBRARY=${LEVEL_ZERO_LIBRARY}
      -DLEVEL_ZERO_INCLUDE_DIR=${LEVEL_ZERO_INCLUDE_DIR}
      -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=TRUE
      -DCYCLES_INCLUDE_DIRS_ESCAPED:STRING=${CORRECT_CYCLES_INCLUDE_DIRS_ESCAPED}&${CMAKE_CURRENT_SOURCE_DIR}/..
      -DORIGINALCMAKE_CXX_COMPILER_ID=${CMAKE_CXX_COMPILER_ID}
      INSTALL_COMMAND ""
      BUILD_ALWAYS TRUE
      STEP_TARGETS build
      EXCLUDE_FROM_ALL TRUE
    )
  else()
    ExternalProject_Add(
      oneapi_kernel
      DOWNLOAD_COMMAND ""
      SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../kernel/device/oneapi
      CMAKE_ARGS
      -DCMAKE_BUILD_TYPE:STRING=${ONEAPI_KERNEL_BUILD_TYPE}
      -DWITH_CYCLES_ONEAPI_BINARIES:BOOL=${WITH_CYCLES_ONEAPI_BINARIES}
      -DCYCLES_ONEAPI_SYCL_TARGET:STRING=${CYCLES_ONEAPI_SYCL_TARGET}
      -DCYCLES_ONEAPI_AOT_TARGETS:STRING=${CYCLES_ONEAPI_AOT_TARGETS}
      -DWITH_CYCLES_ONEAPI_SYCL_HOST_ENABLED:BOOL=${WITH_CYCLES_ONEAPI_SYCL_HOST_ENABLED}
      -DWITH_NANOVDB:BOOL=${WITH_NANOVDB}
      -DNANOVDB_INCLUDE_DIR=${NANOVDB_INCLUDE_DIR}
      -DCMAKE_CXX_COMPILER:STRING=${SYCL_DPCPP_COMPILER}
      -DSYCL_INCLUDE_DIR=${SYCL_INCLUDE_DIR}
      -DSYCL_LIBRARY_DIR=${SYCL_LIBRARY_DIR}
      -DLEVEL_ZERO_LIBRARY=${LEVEL_ZERO_LIBRARY}
      -DLEVEL_ZERO_INCLUDE_DIR=${LEVEL_ZERO_INCLUDE_DIR}
      -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=TRUE
      -DCYCLES_INCLUDE_DIRS_ESCAPED:STRING=${CORRECT_CYCLES_INCLUDE_DIRS_ESCAPED}&${CMAKE_CURRENT_SOURCE_DIR}/..
      -DORIGINALCMAKE_CXX_COMPILER_ID=${CMAKE_CXX_COMPILER_ID}
      INSTALL_COMMAND ""
      BUILD_ALWAYS TRUE
      STEP_TARGETS build
      EXCLUDE_FROM_ALL TRUE
    )
  endif()

  ExternalProject_Get_Property(oneapi_kernel binary_dir)

  if (GENERATOR_IS_MULTI_CONFIG AND CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # make intermediate copy of the DLL since we can't use generator expression in delayed_install
    add_custom_command(TARGET oneapi_kernel-build POST_BUILD
      COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${binary_dir}/$<CONFIG>/${oneapi_kernel_library_filename}"
        "${binary_dir}/${oneapi_kernel_library_filename}"
    )
  endif()

  # The idea of using device as library, is that we compile device cpp files with oneAPI implementation code and then use
  # the code via common C++ interface from .h files. So the rest of the project can be compiled with g++/clang/msbuild
  add_library(cycles_kernel_oneapi SHARED IMPORTED)

  if (GENERATOR_IS_MULTI_CONFIG AND CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    string(REPLACE ".dll" ".lib" oneapi_kernel_library_filename_lib ${oneapi_kernel_library_filename})
    set_target_properties(cycles_kernel_oneapi PROPERTIES
	    POSITION_INDEPENDENT_CODE ON
    )
    foreach(CONFIGURATION IN ITEMS ${CMAKE_CONFIGURATION_TYPES})
 	    string(TOUPPER ${CONFIGURATION} CONFIG)
      set_target_properties(cycles_kernel_oneapi PROPERTIES
 	      IMPORTED_LOCATION_${CONFIG} "${binary_dir}/${CONFIGURATION}/${oneapi_kernel_library_filename}"
 		    IMPORTED_IMPLIB_${CONFIG} "${binary_dir}/${CONFIGURATION}/${oneapi_kernel_library_filename_lib}"
 	    )
    endforeach()
  else()
    set_target_properties(cycles_kernel_oneapi PROPERTIES
        IMPORTED_LOCATION ${binary_dir}/${oneapi_kernel_library_filename}
        POSITION_INDEPENDENT_CODE ON
    )
  endif()

  if (GENERATOR_IS_MULTI_CONFIG AND CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    if(WITH_BLENDER)
      set(CYCLES_DEVICE_ONEAPI_SHARED_LIBRARY_CYCLES_PATH "${binary_dir}/$<CONFIG>/${oneapi_kernel_library_filename}" CACHE INTERNAL "")
    endif()
  else()
    delayed_install(${binary_dir} "${oneapi_kernel_library_filename}" ${CYCLES_INSTALL_PATH}/source/kernel/kernels/oneapi)
    if(WITH_BLENDER)
      set(CYCLES_DEVICE_ONEAPI_SHARED_LIBRARY_CYCLES_PATH "${CYCLES_INSTALL_PATH}/source/kernel/kernels/oneapi/${oneapi_kernel_library_filename}" CACHE INTERNAL "")
    endif()
  endif()

  add_dependencies(cycles_kernel_oneapi oneapi_kernel-build)
endif()
#]]

add_definitions(${GL_DEFINITIONS})

if(WITH_CYCLES_DEVICE_CUDA)
  add_definitions(-DWITH_CUDA)
endif()
if(WITH_CYCLES_DEVICE_HIP)
  add_definitions(-DWITH_HIP)
endif()
if(WITH_CYCLES_DEVICE_OPTIX)
  add_definitions(-DWITH_OPTIX)
endif()
if(WITH_CYCLES_DEVICE_METAL)
  list(APPEND LIB
    ${METAL_LIBRARY}
  )
  add_definitions(-DWITH_METAL)
  list(APPEND SRC
    ${SRC_METAL}
  )
endif()

if(WITH_OPENIMAGEDENOISE)
  list(APPEND LIB
    ${OPENIMAGEDENOISE_LIBRARIES}
  )
endif()

include_directories(${INC})
include_directories(SYSTEM ${INC_SYS})

if (WITH_CYCLES_DEVICE_ONEAPI)
  include_directories(SYSTEM ${SYCL_INCLUDE_DIR}/sycl ${SYCL_INCLUDE_DIR})
  get_filename_component(SYCL_LIBRARY_DIR "${SYCL_LIBRARY}" DIRECTORY)
  if (GENERATOR_IS_MULTI_CONFIG AND CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    link_libraries("${SYCL_LIBRARY_DIR}/sycl$<$<CONFIG:Debug>:d>${CMAKE_LINK_LIBRARY_SUFFIX}")
  else()
    link_libraries(${SYCL_LIBRARY})
  endif()
endif()

cycles_add_library(cycles_device "${LIB}" ${SRC})

if(WITH_CYCLES_DEVICE_ONEAPI)
  # Need to have proper rebuilding in case of changes in cycles_kernel_oneapi due external project behaviour
  add_dependencies(cycles_device cycles_kernel_oneapi)
endif()

source_group("cpu" FILES ${SRC_CPU})
source_group("cuda" FILES ${SRC_CUDA})
source_group("dummy" FILES ${SRC_DUMMY})
source_group("hip" FILES ${SRC_HIP})
source_group("multi" FILES ${SRC_MULTI})
source_group("metal" FILES ${SRC_METAL})
source_group("optix" FILES ${SRC_OPTIX})
source_group("oneapi" FILES ${SRC_ONEAPI})
source_group("common" FILES ${SRC_BASE} ${SRC_HEADERS})
