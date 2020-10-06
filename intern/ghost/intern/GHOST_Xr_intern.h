/*
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */

/** \file
 * \ingroup GHOST
 */

#pragma once

#include <memory>
#include <string.h>
#include <vector>

#include "GHOST_Xr_openxr_includes.h"

#define CHECK_XR(call, error_msg) \
  { \
    XrResult _res = call; \
    if (XR_FAILED(_res)) { \
      throw GHOST_XrException(error_msg, _res); \
    } \
  } \
  (void)0

/**
 * Variation of CHECK_XR() that copies a runtime string to a static error message buffer. Useful
 * when runtime information (e.g. OpenXR action names) should be included in the message.
 */
#define CHECK_XR_BUF(call, error_msg, buf) \
  { \
    XrResult _res = call; \
    if (XR_FAILED(_res)) { \
      strcpy(buf, error_msg); \
      throw GHOST_XrException(buf, _res); \
    } \
  } \
  (void)0

/**
 * Variation of CHECK_XR() that doesn't throw, but asserts for success. Especially useful for
 * destructors, which shouldn't throw.
 */
#define CHECK_XR_ASSERT(call) \
  { \
    XrResult _res = call; \
    assert(_res == XR_SUCCESS); \
    (void)_res; \
  } \
  (void)0

/**
 * Variation of CHECK_XR() that throws but doesn't destroy the runtime. Useful for
 * OpenXR errors (in particular those related to OpenXR actions) that do not warrant
 * aborting the current context / session.
 */
#define CHECK_XR_ND(call, error_msg) \
  { \
    XrResult _res = call; \
    if (XR_FAILED(_res)) { \
      throw GHOST_XrException(error_msg, _res, false); \
    } \
  } \
  (void)0

/**
 * Buffer variation of CHECK_XR_ND().
 */
#define CHECK_XR_ND_BUF(call, error_msg, buf) \
  { \
    XrResult _res = call; \
    if (XR_FAILED(_res)) { \
      strcpy(buf, error_msg); \
      throw GHOST_XrException(buf, _res, false); \
    } \
  } \
  (void)0
