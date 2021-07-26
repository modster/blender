# - Find MessagePack (msgpack) library
# Find the native MessagePack includes and library
# This module defines
#  MSGPACK_INCLUDE_DIRS, where to find spnav.h, Set when
#                        MSGPACK_INCLUDE_DIR is found.
#  MSGPACK_ROOT_DIR, The base directory to search for msgpack.
#                    This can also be an environment variable.
#  MSGPACK_FOUND, If false, do not try to use msgpack.
#
#=============================================================================
# Copyright 2021 Blender Foundation.
#
# Distributed under the OSI-approved BSD 3-Clause License,
# see accompanying file BSD-3-Clause-license.txt for details.
#=============================================================================

# If MSGPACK_ROOT_DIR was defined in the environment, use it.
IF(NOT MSGPACK_ROOT_DIR AND NOT $ENV{MSGPACK_ROOT_DIR} STREQUAL "")
  SET(MSGPACK_ROOT_DIR $ENV{MSGPACK_ROOT_DIR})
ENDIF()

SET(_msgpack_SEARCH_DIRS
  ${MSGPACK_ROOT_DIR}
)

FIND_PATH(MSGPACK_INCLUDE_DIR
  NAMES
    msgpack/include/msgpack.hpp
  HINTS
    ${_msgpack_SEARCH_DIRS}
  PATH_SUFFIXES
    include/msgpack
)

# handle the QUIETLY and REQUIRED arguments and set MSGPACK_FOUND to TRUE if
# all listed variables are TRUE
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(Msgpack DEFAULT_MSG
    MSGPACK_INCLUDE_DIR)

IF(MSGPACK_FOUND)
  SET(MSGPACK_INCLUDE_DIRS ${MSGPACK_INCLUDE_DIR})
ENDIF()

MARK_AS_ADVANCED(
  MSGPACK_INCLUDE_DIR
)

UNSET(_msgpack_SEARCH_DIRS)
