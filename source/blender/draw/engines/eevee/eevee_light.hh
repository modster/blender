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
 * The light module manages light data buffers and light culling system.
 */

#pragma once

#include "BLI_vector.hh"
#include "DNA_light_types.h"

#include "eevee_camera.hh"
#include "eevee_sampling.hh"
#include "eevee_shader.hh"
#include "eevee_shader_shared.hh"
#include "eevee_wrapper.hh"

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

class Light {
 public:
  /* Returns true if the light has any sort of power. */
  static bool sync(LightData &data, const Object *ob, float threshold)
  {
    const ::Light *la = (const ::Light *)ob->data;
    float scale[3];

    float max_power = max_fff(la->r, la->g, la->b) * fabsf(la->energy / 100.0f);
    float surface_max_power = max_ff(la->diff_fac, la->spec_fac) * max_power;
    float volume_max_power = la->volume_fac * max_power;

    data._influence_radius_surface = attenuation_radius_get(la, threshold, surface_max_power);
    data._influence_radius_volume = attenuation_radius_get(la, threshold, volume_max_power);

    /* Cull the light if it has no power. */
    if (max_ff(data._influence_radius_surface, data._influence_radius_volume) == 0.0f) {
      return false;
    }

    mul_v3_v3fl(data.color, &la->r, la->energy);
    normalize_m4_m4_ex(data.object_mat, ob->obmat, scale);
    /* Make sure we have consistent handedness (in case of negatively scaled Z axis). */
    float cross[3];
    cross_v3_v3v3(cross, data._right, data._back);
    if (dot_v3v3(cross, data._up) < 0.0f) {
      negate_v3(data._up);
    }
    shape_parameters_set(data, la, scale);

    float shape_power = shape_power_get(data, la);
    data.diffuse_power = la->diff_fac * shape_power;
    data.specular_power = la->spec_fac * shape_power;
    data.volume_power = la->volume_fac * shape_power_volume_get(data, la);
    data.type = to_light_type(la->type);
    /* No shadow by default */
    data.shadow_id = -1;

    return true;
  }

 private:
  /* Returns attenuation radius inversed & squared for easy bound checking inside the shader. */
  static float attenuation_radius_get(const ::Light *la, float light_threshold, float light_power)
  {
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

  static void shape_parameters_set(LightData &data, const ::Light *la, const float scale[3])
  {
    if (la->type == LA_SPOT) {
      /* Spot size & blend */
      data._spot_scale_x = scale[0] / scale[2];
      data._spot_scale_y = scale[1] / scale[2];
      data.spot_size = cosf(la->spotsize * 0.5f);
      data.spot_blend = (1.0f - data.spot_size) * la->spotblend;
      data.sphere_radius = max_ff(0.001f, la->area_size);
    }
    else if (la->type == LA_AREA) {
      data._area_size_x = max_ff(0.003f, la->area_size * scale[0] * 0.5f);
      if (ELEM(la->area_shape, LA_AREA_RECT, LA_AREA_ELLIPSE)) {
        data._area_size_y = max_ff(0.003f, la->area_sizey * scale[1] * 0.5f);
      }
      else {
        data._area_size_y = max_ff(0.003f, la->area_size * scale[1] * 0.5f);
      }
      /* For volume point lighting. */
      data.sphere_radius = max_ff(0.001f, hypotf(data._area_size_x, data._area_size_y) * 0.5f);
    }
    else if (la->type == LA_SUN) {
      data.sphere_radius = max_ff(0.001f, tanf(min_ff(la->sun_angle, DEG2RADF(179.9f)) / 2.0f));
    }
    else {
      data.sphere_radius = max_ff(0.001f, la->area_size);
    }
  }

  static float shape_power_get(LightData &data, const ::Light *la)
  {
    float power;
    /* Make illumination power constant */
    if (la->type == LA_AREA) {
      float area = data._area_size_x * data._area_size_y;
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
      power = 1.0f / (4.0f * square_f(data.sphere_radius) * float(M_PI * M_PI));
    }
    else { /* LA_SUN */
      power = 1.0f / (square_f(data.sphere_radius) * float(M_PI));
      /* Make illumination power closer to cycles for bigger radii. Cycles uses a cos^3 term that
       * we cannot reproduce so we account for that by scaling the light power. This function is
       * the result of a rough manual fitting. */
      /* Simplification of:
       * power *= 1 + r²/2 */
      power += 1.0f / (2.0f * M_PI);
    }
    return power;
  }

  static float shape_power_volume_get(LightData &data, const ::Light *la)
  {
    /* Volume light is evaluated as point lights. Remove the shape power. */
    if (la->type == LA_AREA) {
      /* Match cycles. Empirical fit... must correspond to some constant. */
      float power = 0.0792f * M_PI;

      /* This corrects for area light most representative point trick. The fit was found by
       * reducing the average error compared to cycles. */
      float area = data._area_size_x * data._area_size_y;
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
};

/** \} */

/* -------------------------------------------------------------------- */
/** \name LightModule
 * \{ */

class LightModule {
 private:
  ShaderModule &shaders_;
  Sampling &sampling_;

  SceneData &scene_data_;

  StructArrayBuffer<LightData, LIGHT_MAX> lights_data_;

  /** Light culling grid. Allocated. */
  /* TODO(fclem) This could become a temp texture once the texture pool supports 3D textures. */
  GPUTexture *light_culling_tx_ = nullptr;

  float light_threshold_;

 public:
  LightModule(ShaderModule &shaders, Sampling &sampling, SceneData &scene_data)
      : shaders_(shaders), sampling_(sampling), scene_data_(scene_data){};

  ~LightModule(){};

  void begin_sync(const Scene *scene)
  {
    /* In begin_sync so it can be aninated. */
    light_threshold_ = max_ff(1e-16f, scene->eevee.light_threshold);
    scene_data_.light_count = 0;
  }

  void sync_light(const Object *ob)
  {
    if (scene_data_.light_count == LIGHT_MAX) {
      return;
    }

    if (Light::sync(lights_data_[scene_data_.light_count], ob, light_threshold_)) {
      scene_data_.light_count++;
    }
  }

  void end_sync(void)
  {
    lights_data_.push_update();
  }

  const GPUUniformBuf *ubo_get(void) const
  {
    return lights_data_.ubo_get();
  }
};

/** \} */

}  // namespace blender::eevee
