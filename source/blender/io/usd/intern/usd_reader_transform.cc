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

#include "usd_reader_transform.h"

#include "BKE_object.h"
#include "DNA_object_types.h"


namespace blender::io::usd {

UsdTransformReader::UsdTransformReader(const pxr::UsdPrim &prim, const USDImporterContext &context)
  : UsdObjectReader(prim, context), xform_(prim)
{

}

bool UsdTransformReader::valid() const
{
  return static_cast<bool>(xform_);
}

void UsdTransformReader::readObjectData(Main *bmain, double time)
{
  if (!this->valid()) {
    return;
  }

  /* Determine prim visibility.
   * TODO(makowalski): Consider optimizations to avoid this expensive call,
   * for example, by determining visibility during stage traversal. */
  pxr::TfToken vis_tok = this->xform_.ComputeVisibility();

  if (vis_tok == pxr::UsdGeomTokens->invisible) {
    return;
  }

  this->object_ = BKE_object_add_only_object(bmain, OB_EMPTY, this->prim_name_.c_str());
  this->object_->data = NULL;
}

}  // namespace blender::io::usd
