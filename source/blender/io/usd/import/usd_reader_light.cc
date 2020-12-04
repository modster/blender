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

#include "usd_reader_light.h"

#include "BLI_assert.h"
#include "BLI_math_base.h"
#include "BLI_math_matrix.h"
#include "BLI_math_rotation.h"
#include "BLI_utildefines.h"

#include "BKE_light.h"
#include "BKE_object.h"

#include "DNA_light_types.h"
#include "DNA_object_types.h"

#include <pxr/usd/usdLux/diskLight.h>
#include <pxr/usd/usdLux/distantLight.h>
#include <pxr/usd/usdLux/rectLight.h>
#include <pxr/usd/usdLux/sphereLight.h>

#include <iostream>

namespace blender::io::usd {

USDLightReader::USDLightReader(const pxr::UsdPrim &prim, const USDImporterContext &context)
    : USDXformableReader(prim, context), light_(prim)
{
}

bool USDLightReader::valid() const
{
  return static_cast<bool>(light_);
}

void USDLightReader::create_object(Main *bmain, double time, USDDataCache *data_cache)
{
  if (!this->valid()) {
    return;
  }

  /* Determine prim visibility.
   * TODO(makowalski): Consider optimizations to avoid this expensive call,
   * for example, by determining visibility during stage traversal. */
  pxr::TfToken vis_tok = this->light_.ComputeVisibility();

  if (vis_tok == pxr::UsdGeomTokens->invisible) {
    return;
  }

  std::string obj_name = get_object_name();

  if (obj_name.empty()) {
    /* Sanity check. */
    std::cerr << "Warning: couldn't determine object name for " << this->prim_path() << std::endl;
  }

  this->object_ = BKE_object_add_only_object(bmain, OB_LAMP, obj_name.c_str());

  std::string light_name = get_data_name();

  if (light_name.empty()) {
    /* Sanity check. */
    std::cerr << "Warning: couldn't determine light name for " << this->prim_path() << std::endl;
  }

  Light *light = (Light *)BKE_light_add(bmain, light_name.c_str());

  this->object_->data = light;

  if (prim_.IsA<pxr::UsdLuxDiskLight>()) {
    light->type = LA_AREA;
    light->area_shape = LA_AREA_DISK;

    pxr::UsdLuxDiskLight disk_light(this->prim_);
    if (disk_light) {
      if (pxr::UsdAttribute radius_attr = disk_light.GetRadiusAttr()) {
        float radius = 1.0;
        radius_attr.Get(&radius, time);
        light->area_size = radius;
      }
    }
  }
  else if (prim_.IsA<pxr::UsdLuxDistantLight>()) {
    light->type = LA_SUN;
  }
  else if (prim_.IsA<pxr::UsdLuxRectLight>()) {
    light->type = LA_AREA;
    light->area_shape = LA_AREA_RECT;

    pxr::UsdLuxRectLight rect_light(this->prim_);
    if (rect_light) {
      if (pxr::UsdAttribute width_attr = rect_light.GetWidthAttr()) {
        float width = 1.0;
        width_attr.Get(&width, time);
        light->area_size = width;
      }
      if (pxr::UsdAttribute height_attr = rect_light.GetHeightAttr()) {
        float height = 1.0;
        height_attr.Get(&height, time);
        light->area_sizey = height;
      }
    }
  }
  else if (prim_.IsA<pxr::UsdLuxSphereLight>()) {
    light->type = LA_LOCAL;

    pxr::UsdLuxSphereLight sphere_light(this->prim_);
    if (sphere_light) {
      if (pxr::UsdAttribute radius_attr = sphere_light.GetRadiusAttr()) {
        float radius = 1.0;
        radius_attr.Get(&radius, time);
        light->area_size = radius;
      }
    }
  }

  if (pxr::UsdAttribute intensity_attr = light_.GetIntensityAttr()) {
    float intensity = 1.0;
    intensity_attr.Get(&intensity, time);
    light->energy = intensity * context_.import_params.light_intensity_scale;

    /* Apply inverse of scaling applied on export in USDLightWriter::do_write().
     * TODO(makowalski): confirm that this scaling is correct. */
    if (light->type != LA_SUN) {
      light->energy *= 100.0f;
    }
  }

  if (pxr::UsdAttribute color_attr = light_.GetColorAttr()) {
    pxr::GfVec3f color(1.0);
    color_attr.Get(&color, time);
    light->r = color[0];
    light->g = color[1];
    light->b = color[2];
  }

  if (pxr::UsdAttribute specular_attr = light_.GetSpecularAttr()) {
    float specular = 1.0f;
    specular_attr.Get(&specular, time);
    light->spec_fac = specular;
  }
}

void USDLightReader::read_matrix(float r_mat[4][4] /* local matrix */,
                                 const double time,
                                 const float scale) const
{
  USDXformableReader::read_matrix(r_mat, time, scale);

  /* Conveting from y-up to z-up requires adjusting
   * the light rotation. */
  if (this->context_.stage_up_axis == pxr::UsdGeomTokens->y) {
    float camera_rotation[4][4];
    axis_angle_to_mat4_single(camera_rotation, 'X', M_PI_2);
    mul_m4_m4m4(r_mat, r_mat, camera_rotation);
  }
}

}  // namespace blender::io::usd
