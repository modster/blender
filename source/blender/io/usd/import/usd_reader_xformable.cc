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

#include "BKE_constraint.h"
#include "BKE_lib_id.h"
#include "BKE_object.h"
#include "DNA_cachefile_types.h"
#include "DNA_constraint_types.h"
#include "DNA_object_types.h"

#include "BLI_listbase.h"
#include "BLI_math_geom.h"
#include "BLI_path_util.h"
#include "BLI_string.h"
#include "BLI_utildefines.h"

#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/matrix4f.h>
#include <pxr/usd/usdGeom/xform.h>
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

void USDXformableReader::eval_merged_with_parent()
{
  merged_with_parent_ = false;
  if (valid() && can_merge_with_parent()) {
    pxr::UsdPrim parent = prim_.GetParent();

    // Merge with the parent if the parent is an Xform and has only one child.
    if (parent && !parent.IsPseudoRoot() && parent.IsA<pxr::UsdGeomXform>()) {

      pxr::UsdPrimSiblingRange child_prims = parent.GetFilteredChildren(
          pxr::UsdTraverseInstanceProxies(pxr::UsdPrimDefaultPredicate));

      // UsdPrimSiblingRange doesn't have a size() function, so we compute the child count.
      std::ptrdiff_t num_child_prims = std::distance(child_prims.begin(), child_prims.end());

      merged_with_parent_ = num_child_prims == 1;
    }
  }
}

void USDXformableReader::set_object_transform(const double time, CacheFile *cache_file)
{
  if (!object_) {
    return;
  }

  float transform_from_usd[4][4];
  bool is_constant = true;

  this->read_matrix(transform_from_usd, time, this->context_.import_params.scale, is_constant);

  /* Apply the matrix to the object. */
  BKE_object_apply_mat4(object_, transform_from_usd, true, false);
  BKE_object_to_mat4(object_, object_->obmat);

  if (cache_file && (!is_constant || this->context_.import_params.is_sequence)) {
    bConstraint *con = BKE_constraint_add_for_object(
        object_, NULL, CONSTRAINT_TYPE_TRANSFORM_CACHE);
    bTransformCacheConstraint *data = static_cast<bTransformCacheConstraint *>(con->data);
    BLI_strncpy(data->object_path, this->prim_path().c_str(), FILE_MAX);

    data->cache_file = cache_file;
    id_us_plus(&data->cache_file->id);
  }
}

void USDXformableReader::read_matrix(float r_mat[4][4] /* local matrix */,
                                     const double time,
                                     const float scale,
                                     bool &is_constant) const
{
  is_constant = true;

  pxr::UsdGeomXformable xformable(prim_);

  if (!xformable) {
    unit_m4(r_mat);
    return;
  }

  is_constant = !xformable.TransformMightBeTimeVarying();

  pxr::GfMatrix4d usd_local_xf;
  bool reset_xform_stack;
  xformable.GetLocalTransformation(&usd_local_xf, &reset_xform_stack, time);

  if (merged_with_parent_) {
    /* Take into account the parent's local xform. */
    pxr::UsdGeomXformable parent_xformable(prim_.GetParent());

    if (parent_xformable) {
      is_constant = is_constant && !parent_xformable.TransformMightBeTimeVarying();

      pxr::GfMatrix4d usd_parent_local_xf;
      parent_xformable.GetLocalTransformation(&usd_parent_local_xf, &reset_xform_stack, time);

      usd_local_xf = usd_local_xf * usd_parent_local_xf;
    }
  }

  // Convert the result to a float matrix.
  pxr::GfMatrix4f mat4f = pxr::GfMatrix4f(usd_local_xf);
  mat4f.Get(r_mat);

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
