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

#include "BLI_float3.hh"
#include "BLI_listbase.h"
#include "BLI_math.h"

#include "DEG_depsgraph.h"
#include "DEG_depsgraph_query.h"

#include "IO_wavefront_obj.h"
#include "obj_export_nurbs.hh"

namespace blender::io::obj {
OBJCurve::OBJCurve(Depsgraph *depsgraph,
                   const OBJExportParams &export_params,
                   Object *export_object)
    : export_object_eval_(export_object)
{
  export_object_eval_ = DEG_get_evaluated_object(depsgraph, export_object);
  export_curve_ = static_cast<Curve *>(export_object_eval_->data);
  set_world_axes_transform(export_params.forward_axis, export_params.up_axis);
}

/**
 * Store the product of export axes settings and an object's world transform
 * matrix.
 */
void OBJCurve::set_world_axes_transform(const eTransformAxisForward forward,
                                        const eTransformAxisUp up)
{
  float axes_transform[3][3];
  unit_m3(axes_transform);
  /* -Y-forward and +Z-up are the default Blender axis settings. */
  mat3_from_axis_conversion(
      OBJ_AXIS_NEGATIVE_Y_FORWARD, OBJ_AXIS_Z_UP, forward, up, axes_transform);
  mul_m4_m3m4(world_axes_transform_, axes_transform, export_object_eval_->obmat);
  /* mul_m4_m3m4 does not copy last row of obmat, i.e. location data. */
  copy_v4_v4(world_axes_transform_[3], export_object_eval_->obmat[3]);
}

const char *OBJCurve::get_curve_name() const
{
  return export_object_eval_->id.name + 2;
}

int OBJCurve::tot_nurbs() const
{
  return BLI_listbase_count(&export_curve_->nurb);
}

int OBJCurve::get_nurbs_points(const int index) const
{
  const Nurb *nurb = static_cast<Nurb *>(BLI_findlink(&export_curve_->nurb, index));
  return nurb->pntsu * nurb->pntsv;
}

/**
 * Get coordinates of a vertex at the given index.
 */
float3 OBJCurve::calc_nurbs_point_coords(const int index,
                                         const int vert_index,
                                         const float scaling_factor) const
{
  const Nurb *nurb = static_cast<Nurb *>(BLI_findlink(&export_curve_->nurb, index));
  float3 r_coord;
  const BPoint &bpoint = nurb->bp[vert_index];
  copy_v3_v3(r_coord, bpoint.vec);
  mul_m4_v3(world_axes_transform_, r_coord);
  mul_v3_fl(r_coord, scaling_factor);
  return r_coord;
}

/**
 * Get number of "curv" points of the Nurb at the given index.
 */
int OBJCurve::get_nurbs_num(const int index) const
{
  const Nurb *nurb = static_cast<Nurb *>(BLI_findlink(&export_curve_->nurb, index));
  const int r_nurbs_degree = nurb->orderu - 1;
  /* "curv_num" is the number of control points in a nurbs.
   * If it is cyclic, degree also adds up. */
  int r_curv_num = nurb->pntsv * nurb->pntsu;
  if (nurb->flagu & CU_NURB_CYCLIC) {
    r_curv_num += r_nurbs_degree;
  }
  return r_curv_num;
}

/**
 * Get the degree of the Nurb at the given index.
 */
int OBJCurve::get_nurbs_degree(const int index) const
{
  const Nurb *nurb = static_cast<Nurb *>(BLI_findlink(&export_curve_->nurb, index));
  return nurb->orderu - 1;
}

}  // namespace blender::io::obj
