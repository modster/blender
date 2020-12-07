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

#include "BLI_math_base.h"
#include "BLI_math_matrix.h"
#include "BLI_math_rotation.h"

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

void USDCameraReader::create_object(Main *bmain, double time, USDDataCache *data_cache)
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

  /* TODO(makowalski): The following application of
   * settings is taken from the ABC importer.  Verify
   * that this logic makes sense for USD. */

  Camera *bcam = static_cast<Camera *>(BKE_camera_add(bmain, cam_name.c_str()));

  pxr::GfCamera usd_cam = camera_.GetCamera(time);

  const float apperture_x = usd_cam.GetHorizontalAperture();
  const float apperture_y = usd_cam.GetVerticalAperture();
  const float h_film_offset = usd_cam.GetHorizontalApertureOffset();
  const float v_film_offset = usd_cam.GetVerticalApertureOffset();
  const float film_aspect = apperture_x / apperture_y;

  bcam->type = usd_cam.GetProjection() == pxr::GfCamera::Perspective ? CAM_PERSP : CAM_ORTHO;

  bcam->lens = usd_cam.GetFocalLength();

  bcam->sensor_x = apperture_x;
  bcam->sensor_y = apperture_y;

  bcam->shiftx = h_film_offset / apperture_x;
  bcam->shifty = v_film_offset / apperture_y / film_aspect;

  pxr::GfRange1f usd_clip_range = usd_cam.GetClippingRange();
  bcam->clip_start = usd_clip_range.GetMin();
  bcam->clip_end = usd_clip_range.GetMax();

  bcam->dof.focus_distance = usd_cam.GetFocusDistance();
  bcam->dof.aperture_fstop = usd_cam.GetFStop();

  this->object_->data = bcam;
}

void USDCameraReader::read_matrix(float r_mat[4][4] /* local matrix */,
                                  const double time,
                                  const float scale,
                                  bool &is_constant) const
{
  USDXformableReader::read_matrix(r_mat, time, scale, is_constant);

  /* Conveting from y-up to z-up requires adjusting
   * the camera rotation. */
  if (this->context_.stage_up_axis == pxr::UsdGeomTokens->y) {
    float camera_rotation[4][4];
    axis_angle_to_mat4_single(camera_rotation, 'X', M_PI_2);
    mul_m4_m4m4(r_mat, r_mat, camera_rotation);
  }
}

}  // namespace blender::io::usd
