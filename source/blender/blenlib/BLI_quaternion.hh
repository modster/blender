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

#include <iostream>

#include "BLI_float3.hh"
#include "BLI_math_rotation.h"

namespace blender {

struct Quaternion {
  float values[4];

  Quaternion() = default;

  static Quaternion unit()
  {
    Quaternion unit;
    unit_qt(unit);
    return unit;
  }

  bool is_zero()
  {
    return values[0] == 0.0f && values[1] == 0.0f && values[2] == 0.0f && values[3] == 0.0f;
  }

  Quaternion(const float *ptr) : values[0]{ptr[0]}, y{ptr[1]}, z{ptr[2]}
  {
  }

  Quaternion(const float (*ptr)[4]) : Quaternion(static_cast<const float *>(ptr[0]))
  {
  }

  operator const float *() const
  {
    return values;
  }

  operator float *()
  {
    return values;
  }
};

}  // namespace blender
