# - Find SHADERC library
# Find the native Haru includes and library
# This module defines
#  SHADERC_INCLUDE_DIRS, where to find hpdf.h, set when
#                        SHADERC_INCLUDE_DIR is found.
#  SHADERC_LIBRARIES, libraries to link against to use Haru.
#  SHADERC_ROOT_DIR, The base directory to search for Haru.
#                    This can also be an environment variable.
#  SHADERC_FOUND, If false, do not try to use Haru.
#
# also defined, but not for general use are
#  SHADERC_LIBRARY, where to find the Haru library.

#=============================================================================
# Copyright 2021 Blender Foundation.
#
# Distributed under the OSI-approved BSD 3-Clause License,
# see accompanying file BSD-3-Clause-license.txt for details.
#=============================================================================

# If SHADERC_ROOT_DIR was defined in the environment, use it.
if(NOT SHADERC_ROOT_DIR AND NOT $ENV{SHADERC_ROOT_DIR} STREQUAL "")
  set(SHADERC_ROOT_DIR $ENV{SHADERC_ROOT_DIR})
endif()

set(_shaderc_SEARCH_DIRS
  ${SHADERC_ROOT_DIR}
  /opt/lib/haru
)

find_path(SHADERC_INCLUDE_DIR
  NAMES
    shaderc.hpp
  HINTS
    ${_shaderc_SEARCH_DIRS}
  PATH_SUFFIXES
    include/shaderc
    include
)

find_library(SHADERC_LIBRARY
  NAMES
    shaderc_combined
    shaderc
  HINTS
    ${_shaderc_SEARCH_DIRS}
  PATH_SUFFIXES
    lib64 lib
)

# Handle the QUIETLY and REQUIRED arguments and set SHADERC_FOUND to TRUE if
# all listed variables are TRUE.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ShaderC DEFAULT_MSG SHADERC_LIBRARY SHADERC_INCLUDE_DIR)

if(SHADERC_FOUND)
  set(SHADERC_LIBRARIES ${SHADERC_LIBRARY})
  set(SHADERC_INCLUDE_DIRS ${SHADERC_INCLUDE_DIR})
endif()

mark_as_advanced(
  SHADERC_INCLUDE_DIR
  SHADERC_LIBRARY
)

unset(_shaderc_SEARCH_DIRS)
