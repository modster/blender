# SPDX-License-Identifier: BSD-3-Clause
# Copyright 2021-2022 Intel Corporation

# - Find SYCL library
# Find the native SYCL header and libraries needed by oneAPI implementation
# This module defines
#  SYCL_DPCPP_COMPILER, compiler from oneAPI toolkit, which will be used for compilation of SYCL code
#  SYCL_LIBRARY, libraries to link against in order to use SYCL.
#  SYCL_INCLUDE_DIR, directories where SYCL headers can be found
#  SYCL_ROOT_DIR, The base directory to search for SYCL files.
#                 This can also be an environment variable.
#  SYCL_FOUND, If false, then don't try to use SYCL.

IF(NOT SYCL_ROOT_DIR AND NOT $ENV{SYCL_ROOT_DIR} STREQUAL "")
  SET(SYCL_ROOT_DIR $ENV{SYCL_ROOT_DIR})
ENDIF()

SET(_sycl_search_dirs
  ${SYCL_ROOT_DIR}
  /usr/lib
  /usr/local/lib
  /opt/intel/oneapi/compiler/latest/linux/
  C:/Program\ Files\ \(x86\)/Intel/oneAPI/compiler/latest/windows
)

FIND_PROGRAM(SYCL_DPCPP_COMPILER
  NAMES
    dpcpp
  HINTS
    ${_sycl_search_dirs}
  PATH_SUFFIXES
    bin
)

FIND_LIBRARY(_SYCL_LIBRARY
  NAMES
    sycl
  HINTS
    ${_sycl_search_dirs}
  PATH_SUFFIXES
    lib64 lib
)

FIND_PATH(_SYCL_INCLUDE_DIR
  NAMES
    CL/sycl.hpp
  HINTS
    ${_sycl_search_dirs}
  PATH_SUFFIXES
    include
    include/sycl
)

INCLUDE(FindPackageHandleStandardArgs)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(SYCL DEFAULT_MSG _SYCL_LIBRARY _SYCL_INCLUDE_DIR)

IF(SYCL_FOUND)
  SET(SYCL_LIBRARY ${_SYCL_LIBRARY})

  get_filename_component(_SYCL_INCLUDE_PARENT_DIR ${_SYCL_INCLUDE_DIR} DIRECTORY)

  SET(SYCL_INCLUDE_DIR ${_SYCL_INCLUDE_DIR} ${_SYCL_INCLUDE_PARENT_DIR})
ELSE()
  SET(SYCL_SYCL_FOUND FALSE)
ENDIF()

MARK_AS_ADVANCED(
  SYCL_LIBRARY
  SYCL_INCLUDE_DIR
  SYCL_DPCPP_COMPILER
  _SYCL_INCLUDE_DIR
  _SYCL_INCLUDE_PARENT_DIR
  _SYCL_LIBRARY
)
