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
 * Copyright 2021, Blender Foundation.
 */

/** \file
 * \ingroup eevee
 *
 * An instance contains all structures needed to do a complete render.
 */

#include "eevee_instance.hh"

#include "eevee_light.hh"

namespace blender::eevee {

/* -------------------------------------------------------------------- */
/** \name LightData
 * \{ */

static eLightType to_light_type(short blender_light_type)
{
  switch (blender_light_type) {
    default:
    case LA_LOCAL:
      return LIGHT_POINT;
    case LA_SUN:
      return LIGHT_SUN;
    case LA_SPOT:
      return LIGHT_SPOT;
    case LA_AREA:
      return ELEM(blender_light_type, LA_AREA_DISK, LA_AREA_ELLIPSE) ? LIGHT_ELIPSE : LIGHT_RECT;
  }
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Light Object
 * \{ */

Light::Light(const Object *ob, float threshold)
{
  const ::Light *la = (const ::Light *)ob->data;
  float scale[3];

  float max_power = max_fff(la->r, la->g, la->b) * fabsf(la->energy / 100.0f);
  float surface_max_power = max_ff(la->diff_fac, la->spec_fac) * max_power;
  float volume_max_power = la->volume_fac * max_power;

  this->inv_sqr_influence_radius_surface = inverse_squared_attenuation_radius_get(
      la, threshold, surface_max_power);
  this->inv_sqr_influence_radius_volume = inverse_squared_attenuation_radius_get(
      la, threshold, volume_max_power);

  mul_v3_v3fl(this->color, &la->r, la->energy);
  normalize_m4_m4_ex(this->object_mat, ob->obmat, scale);
  /* Make sure we have consistent handedness (in case of negatively scaled Z axis). */
  float cross[3];
  cross_v3_v3v3(cross, this->_right, this->_back);
  if (dot_v3v3(cross, this->_up) < 0.0f) {
    negate_v3(this->_up);
  }

  shape_parameters_set(la, scale);

  float shape_power = shape_power_get(la);
  this->diffuse_power = la->diff_fac * shape_power;
  this->specular_power = la->spec_fac * shape_power;
  this->volume_power = la->volume_fac * shape_power_volume_get(la);
  this->type = to_light_type(la->type);
  /* No shadow by default */
  this->shadow_id = -1;
}

/* Returns attenuation radius inversed & squared for easy bound checking inside the shader. */
float Light::inverse_squared_attenuation_radius_get(const ::Light *la,
                                                    float light_threshold,
                                                    float light_power)
{
  if (la->type == LA_SUN) {
    return (light_power > 1e-5f) ? 1e16f : 0.0f;
  }

  if (la->mode & LA_CUSTOM_ATTENUATION) {
    return la->att_dist;
  }
  /* Compute the distance (using the inverse square law)
   * at which the light power reaches the light_threshold. */
  /* TODO take area light scale into account. */
#if 1 /* Optimized. */
  return (light_power > 1e-5f) ? (light_threshold / light_power) : 0.0f;
#else
    float radius = sqrtf(light_power / light_threshold));
    return 1.0f / max_ff(1e-4f, square_f(radius));
#endif
}

void Light::shape_parameters_set(const ::Light *la, const float scale[3])
{
  if (la->type == LA_SPOT) {
    /* Spot size & blend */
    this->_spot_scale_x = scale[0] / scale[2];
    this->_spot_scale_y = scale[1] / scale[2];
    this->_spot_size = cosf(la->spotsize * 0.5f);
    this->_spot_blend = (1.0f - this->_spot_size) * la->spotblend;
    sphere_radius = max_ff(0.001f, la->area_size);
  }
  else if (la->type == LA_AREA) {
    float area_size_y = (ELEM(la->area_shape, LA_AREA_RECT, LA_AREA_ELLIPSE)) ? la->area_sizey :
                                                                                la->area_size;
    this->_area_size_x = max_ff(0.003f, la->area_size * scale[0] * 0.5f);
    this->_area_size_y = max_ff(0.003f, area_size_y * scale[1] * 0.5f);
    /* For volume point lighting. */
    sphere_radius = max_ff(0.001f, hypotf(_area_size_x, _area_size_y) * 0.5f);
  }
  else if (la->type == LA_SUN) {
    sphere_radius = max_ff(0.001f, tanf(min_ff(la->sun_angle, DEG2RADF(179.9f)) / 2.0f));
  }
  else {
    sphere_radius = max_ff(0.001f, la->area_size);
  }
}

float Light::shape_power_get(const ::Light *la)
{
  float power;
  /* Make illumination power constant */
  if (la->type == LA_AREA) {
    float area = _area_size_x * _area_size_y;
    power = 1.0f / (area * 4.0f * float(M_PI));
    /* FIXME : Empirical, Fit cycles power */
    power *= 0.8f;
    if (ELEM(la->area_shape, LA_AREA_DISK, LA_AREA_ELLIPSE)) {
      /* Scale power to account for the lower area of the ellipse compared to the surrounding
       * rectangle. */
      power *= 4.0f / M_PI;
    }
  }
  else if (ELEM(la->type, LA_SPOT, LA_LOCAL)) {
    power = 1.0f / (4.0f * square_f(this->sphere_radius) * float(M_PI * M_PI));
  }
  else { /* LA_SUN */
    power = 1.0f / (square_f(this->sphere_radius) * float(M_PI));
    /* Make illumination power closer to cycles for bigger radii. Cycles uses a cos^3 term that
     * we cannot reproduce so we account for that by scaling the light power. This function is
     * the result of a rough manual fitting. */
    /* Simplification of:
     * power *= 1 + r²/2 */
    power += 1.0f / (2.0f * M_PI);
  }
  return power;
}

float Light::shape_power_volume_get(const ::Light *la)
{
  /* Volume light is evaluated as point lights. Remove the shape power. */
  if (la->type == LA_AREA) {
    /* Match cycles. Empirical fit... must correspond to some constant. */
    float power = 0.0792f * M_PI;

    /* This corrects for area light most representative point trick. The fit was found by
     * reducing the average error compared to cycles. */
    float area = this->_area_size_x * this->_area_size_y;
    float tmp = M_PI_2 / (M_PI_2 + sqrtf(area));
    /* Lerp between 1.0 and the limit (1 / pi). */
    power *= tmp + (1.0f - tmp) * M_1_PI;

    return power;
  }
  else if (ELEM(la->type, LA_SPOT, LA_LOCAL)) {
    /* Match cycles. Empirical fit... must correspond to some constant. */
    return 0.0792f;
  }
  else { /* LA_SUN */
    return 1.0f;
  }
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name LightModule
 * \{ */

void LightModule::begin_sync(void)
{
  /* In begin_sync so it can be aninated. */
  light_threshold_ = max_ff(1e-16f, inst_.scene->eevee.light_threshold);

  lights_.clear();
}

void LightModule::sync_light(const Object *ob)
{
  lights_.append(eevee::Light(ob, light_threshold_));
}

void LightModule::end_sync(void)
{
}

void LightModule::bind_range(int range_id)
{
  int64_t range_start = range_id * LIGHT_MAX;
  int64_t range_size = min_ii(this->lights_.size() - range_start, LIGHT_MAX);
  /* TODO(fclem) Can we avoid this copy in the first iter? */
  LightData *src = lights_.data() + range_start;
  LightData *dst = lights_data_.data();
  memcpy(dst, src, sizeof(LightData) * range_size);
  lights_data_.push_update();
  clusters_data_.light_count = range_size;
  clusters_data_.push_update();
}

/** \} */

}  // namespace blender::eevee
