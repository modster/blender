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

#include "BLI_compiler_compat.h"

#include "pxr/usd/usd/common.h"

#include <vector>

namespace blender::io::usd {

/* TODO(makowalski):  copy_m44_axis_swap, create_swapped_rotation_matrix
 * and copy_zup_from_yup below are duplicates of the declarations in
 * abc_axis_conversion.h, and should be moved to a shared location. */
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

BLI_INLINE void copy_zup_from_yup(float zup[3], const float yup[3])
{
  const float old_yup1 = yup[1]; /* in case zup == yup */
  zup[0] = yup[0];
  zup[1] = -yup[2];
  zup[2] = old_yup1;
}

} /* namespace blender::io::usd */
