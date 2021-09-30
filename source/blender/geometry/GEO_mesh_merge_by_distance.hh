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

#pragma once

/** \file
 * \ingroup geo
 */

#include "BLI_span.hh"

namespace blender::geometry {

enum class WeldMode {
  all = 0,
  connected = 1,
};

WeldMode weld_mode_from_int(const int16_t type);
int16_t weld_mode_to_int(const WeldMode weld_mode);

struct Mesh *mesh_merge_by_distance(struct Mesh *mesh,
                                    const Span<bool> mask,
                                    const float merge_distance,
                                    const WeldMode weld_mode);

}  // namespace blender::geometry
