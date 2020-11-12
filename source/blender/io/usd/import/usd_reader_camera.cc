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

#include "usd_reader_camera.h"

#include "DNA_camera_types.h"
#include "DNA_object_types.h"

#include "BKE_camera.h"
#include "BKE_object.h"

#include <iostream>

namespace blender::io::usd {

USDCameraReader::USDCameraReader(const pxr::UsdPrim &prim, const USDImporterContext &context)
    : USDXformableReader(prim, context), camera_(prim)
{
}

bool USDCameraReader::valid() const
{
  return static_cast<bool>(camera_);
}

void USDCameraReader::create_object(Main *bmain, double time)
{
  if (!this->valid()) {
    return;
  }

  /* Determine prim visibility.
   * TODO(makowalski): Consider optimizations to avoid this expensive call,
   * for example, by determining visibility during stage traversal. */
  pxr::TfToken vis_tok = this->camera_.ComputeVisibility();

  if (vis_tok == pxr::UsdGeomTokens->invisible) {
    return;
  }

  std::string obj_name = get_object_name();

  if (obj_name.empty()) {
    /* Sanity check. */
    std::cerr << "Warning: couldn't determine object name for " << this->prim_path() << std::endl;
  }

  this->object_ = BKE_object_add_only_object(bmain, OB_CAMERA, obj_name.c_str());

  std::string cam_name = get_data_name();

  if (cam_name.empty()) {
    /* Sanity check. */
    std::cerr << "Warning: couldn't determine camera name for " << this->prim_path() << std::endl;
  }

  Camera *bcam = static_cast<Camera *>(BKE_camera_add(bmain, cam_name.c_str()));

  pxr::GfCamera usd_cam = camera_.GetCamera(time);

  bcam->lens = usd_cam.GetFocalLength();

  pxr::GfRange1f usd_clip_range = usd_cam.GetClippingRange();

  bcam->clip_start = usd_clip_range.GetMin();
  bcam->clip_end = usd_clip_range.GetMax();

  this->object_->data = bcam;
}

}  // namespace blender::io::usd
