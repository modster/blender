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
 *
 * The Original Code is Copyright (C) 2020 Blender Foundation.
 * All rights reserved.
 */
#pragma once

#include "pxr/usd/usd/common.h"

#include <vector>

struct USDImporterContext;

namespace blender::io::usd {

class UsdObjectReader;

void debug_traverse_stage(const pxr::UsdStageRefPtr &usd_stage);

// TODO:  This is a duplicate of the definition in abc_util.h.  Should
// move this to a shared location.
void split(const std::string &s, const char delim, std::vector<std::string> &tokens);

// TOD:  copy_m44_axis_swap and create_swapped_rotation_matrix
// below are duplicates of the declarations in abc_axis_conversion.h.
// Should move this to a shared location.
typedef enum {
  USD_ZUP_FROM_YUP = 1,
  USD_YUP_FROM_ZUP = 2,
} UsdAxisSwapMode;


/* Create a rotation matrix for each axis from euler angles.
 * Euler angles are swapped to change coordinate system. */
void create_swapped_rotation_matrix(float rot_x_mat[3][3],
                                    float rot_y_mat[3][3],
                                    float rot_z_mat[3][3],
                                    const float euler[3],
                                    UsdAxisSwapMode mode);

void copy_m44_axis_swap(float dst_mat[4][4], float src_mat[4][4], UsdAxisSwapMode mode);

void create_readers(const pxr::UsdStageRefPtr &usd_stage, std::vector<UsdObjectReader *> &r_readers, const USDImporterContext &context);

} // namespace blender::io::usd
