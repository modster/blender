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
 * \ingroup DNA
 */

#pragma once

/* Struct members on own line. */
/* clang-format off */

/* -------------------------------------------------------------------- */
/** \name CacheFile Struct
 * \{ */

#define _DNA_DEFAULT_CacheFile \
  { \
    .filepath[0] = '\0', \
    .override_frame = false, \
    .frame = 0.0f, \
    .is_sequence = false, \
    .scale = 1.0f, \
    .object_paths ={NULL, NULL}, \
 \
    .handle = NULL, \
    .handle_filepath[0] = '\0', \
    .handle_readers = NULL, \
    .default_radius = 0.01f, \
    .cache_method = CACHEFILE_CACHE_ALL_DATA, \
    .cache_memory_limit = 1024, \
    .cache_frame_count = 10, \
  }

/** \} */

/* clang-format on */
