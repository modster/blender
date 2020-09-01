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

/** \file
 * \ingroup obj
 */

#pragma once

#include "BLI_utility_mixins.hh"

#include "DNA_curve_types.h"

namespace blender::io::obj {
class OBJCurve : NonMovable, NonCopyable {
 private:
  const Depsgraph *depsgraph_;
  const Object *export_object_eval_;
  const Curve *export_curve_;
  float world_axes_transform_[4][4];

 public:
  OBJCurve(Depsgraph *depsgraph, const OBJExportParams &export_params, Object *export_object);

  const char *get_curve_name() const;
  int tot_nurbs() const;
  int get_nurbs_points(const int index) const;
  float3 calc_nurbs_point_coords(const int index,
                                 const int vert_index,
                                 const float scaling_factor) const;
  int get_nurbs_num(const int index) const;
  int get_nurbs_degree(const int index) const;

 private:
  void store_world_axes_transform(const eTransformAxisForward forward, const eTransformAxisUp up);
};

}  // namespace blender::io::obj
