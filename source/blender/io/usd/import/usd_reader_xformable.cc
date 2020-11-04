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

#include "usd_reader_xformable.h"
#include "usd_import_util.h"

#include "BKE_lib_id.h"
#include "BKE_object.h"
#include "DNA_object_types.h"

#include "BLI_listbase.h"
#include "BLI_math_geom.h"
#include "BLI_string.h"
#include "BLI_utildefines.h"

#include <pxr/base/gf/matrix4d.h>
#include <pxr/usd/usdGeom/xformable.h>

#include <iostream>

namespace blender::io::usd {

USDXformableReader::USDXformableReader(const pxr::UsdPrim &prim, const USDImporterContext &context)
    : USDPrimReader(prim, context), object_(nullptr), parent_(nullptr), merged_with_parent_(false)
{
}

USDXformableReader::~USDXformableReader()
{
}

Object *USDXformableReader::object() const
{
  return object_;
}

void USDXformableReader::setup_object_transform(const double time)
{
  if (!this->object_) {
    return;
  }

  bool is_constant = false;
  float transform_from_usd[4][4];

  this->read_matrix(transform_from_usd, time, this->context_.import_params.scale, is_constant);

  /* Apply the matrix to the object. */
  BKE_object_apply_mat4(object_, transform_from_usd, true, false);
  BKE_object_to_mat4(object_, object_->obmat);

  /* TODO(makowalski):  Set up transform constraint if not constant. */
}

void USDXformableReader::read_matrix(float r_mat[4][4] /* local matrix */,
                                     const double time,
                                     const float scale,
                                     bool &is_constant)
{
  pxr::UsdGeomXformable xformable(prim_);

  if (!xformable) {
    unit_m4(r_mat);
    is_constant = true;
    return;
  }

  /* TODO(makowalski):  Check for constant transform. */

  pxr::GfMatrix4d usd_local_xf;
  bool reset_xform_stack;
  xformable.GetLocalTransformation(&usd_local_xf, &reset_xform_stack, time);

  if (merged_with_parent_) {
    /* Take into account the parent's local xform. */
    pxr::UsdGeomXformable parent_xformable(prim_.GetParent());

    if (parent_xformable) {
      pxr::GfMatrix4d usd_parent_local_xf;
      parent_xformable.GetLocalTransformation(&usd_parent_local_xf, &reset_xform_stack, time);

      usd_local_xf = usd_local_xf * usd_parent_local_xf;
    }
  }

  double double_mat[4][4];
  usd_local_xf.Get(double_mat);

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      r_mat[i][j] = static_cast<float>(double_mat[i][j]);
    }
  }

  if (this->context_.stage_up_axis == pxr::UsdGeomTokens->y) {
    /* Swap the matrix from y-up to z-up. */
    copy_m44_axis_swap(r_mat, r_mat, USD_ZUP_FROM_YUP);
  }

  /* Apply scaling only to root objects, parenting will propagate it. */
  if (!this->parent_) {
    float scale_mat[4][4];
    scale_m4_fl(scale_mat, scale);
    mul_m4_m4m4(r_mat, scale_mat, r_mat);
  }
}

} /* namespace blender::io::usd */
