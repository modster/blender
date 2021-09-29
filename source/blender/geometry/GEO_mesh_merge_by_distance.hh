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

namespace blender::geometry {

#ifdef __cplusplus
extern "C" {
#endif

enum class WeldMode {
  all = 0,
  connected = 1,
};

WeldMode GEO_weld_mode_from_int(const short type);
int16_t GEO_weld_mode_to_short(const WeldMode weld_mode);

struct Mesh *GEO_mesh_merge_by_distance(struct Mesh *mesh,
                                        const bool *mask,
                                        const float merge_distance,
                                        const WeldMode weld_mode);
#ifdef __cplusplus
}
#endif

}  // namespace blender::geometry
