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

#include "usd_reader_shape.h"

#include "BKE_object.h"
#include "DNA_object_types.h"

#include <pxr/usd/usdGeom/cube.h>
#include <pxr/usd/usdGeom/gprim.h>
#include <pxr/usd/usdGeom/imageable.h>
#include <pxr/usd/usdGeom/sphere.h>

#include <iostream>

namespace blender::io::usd {

USDShapeReader::USDShapeReader(const pxr::UsdPrim &prim, const USDImporterContext &context)
    : USDXformableReader(prim, context)
{
}

USDShapeReader::~USDShapeReader()
{
}

bool USDShapeReader::valid() const
{
  return static_cast<bool>(pxr::UsdGeomGprim(prim_));
}

void USDShapeReader::create_object(Main *bmain, double time, USDDataCache *data_cache)
{
  if (!this->valid()) {
    return;
  }

  /* Determine prim visibility.
   * TODO(makowalski): Consider optimizations to avoid this expensive call,
   * for example, by determining visibility during stage traversal. */
  pxr::UsdGeomImageable imageable(this->prim_);

  if (!imageable) {
    return;  // Should never happen.
  }

  pxr::TfToken vis_tok = imageable.ComputeVisibility();

  if (vis_tok == pxr::UsdGeomTokens->invisible) {
    return;
  }

  this->object_ = BKE_object_add_only_object(bmain, OB_EMPTY, this->prim_name().c_str());

  if (this->prim_.IsA<pxr::UsdGeomCube>()) {
    this->object_->empty_drawtype = OB_CUBE;

    pxr::UsdGeomCube cube(this->prim_);

    if (cube) {
      double size = 1.0;
      cube.GetSizeAttr().Get<double>(&size);
      this->object_->empty_drawsize = static_cast<float>(0.5 * size);
    }
  }
  else if (this->prim_.IsA<pxr::UsdGeomSphere>()) {
    this->object_->empty_drawtype = OB_EMPTY_SPHERE;

    pxr::UsdGeomSphere sphere(this->prim_);

    if (sphere) {
      double rad = 1.0;
      sphere.GetRadiusAttr().Get<double>(&rad);
      this->object_->empty_drawsize = static_cast<float>(rad);
    }
  }

  this->object_->data = nullptr;
}

}  // namespace blender::io::usd
