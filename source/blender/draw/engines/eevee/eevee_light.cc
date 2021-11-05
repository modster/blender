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

#include "eevee_instance.hh"

#include "eevee_light.hh"

namespace blender::eevee {

/* -------------------------------------------------------------------- */
/** \name LightData
 * \{ */

static eLightType to_light_type(short blender_light_type, short blender_area_type)
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
      return ELEM(blender_area_type, LA_AREA_DISK, LA_AREA_ELLIPSE) ? LIGHT_ELLIPSE : LIGHT_RECT;
  }
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Light Object
 * \{ */

void Light::sync(ShadowModule &shadows, const Object *ob, float threshold)
{
  const ::Light *la = (const ::Light *)ob->data;
  float scale[3];

  float max_power = max_fff(la->r, la->g, la->b) * fabsf(la->energy / 100.0f);
  float surface_max_power = max_ff(la->diff_fac, la->spec_fac) * max_power;
  float volume_max_power = la->volume_fac * max_power;

  float influence_radius_surface = attenuation_radius_get(la, threshold, surface_max_power);
  float influence_radius_volume = attenuation_radius_get(la, threshold, volume_max_power);

  this->influence_radius_max = max_ff(influence_radius_surface, influence_radius_volume);
  this->influence_radius_invsqr_surface = (influence_radius_surface > 1e-8f) ?
                                              (1.0f / square_f(influence_radius_surface)) :
                                              0.0f;
  this->influence_radius_invsqr_volume = (influence_radius_volume > 1e-8f) ?
                                             (1.0f / square_f(influence_radius_volume)) :
                                             0.0f;

  this->color = vec3(&la->r) * la->energy;
  normalize_m4_m4_ex(this->object_mat, ob->obmat, scale);
  /* Make sure we have consistent handedness (in case of negatively scaled Z axis). */
  vec3 cross = vec3::cross(this->_right, this->_up);
  if (vec3::dot(cross, this->_back) < 0.0f) {
    negate_v3(this->_up);
  }

  shape_parameters_set(la, scale);

  float shape_power = shape_power_get(la);
  float point_power = point_power_get(la);
  this->diffuse_power = la->diff_fac * shape_power;
  this->transmit_power = la->diff_fac * point_power;
  this->specular_power = la->spec_fac * shape_power;
  this->volume_power = la->volume_fac * point_power;

  eLightType new_type = to_light_type(la->type, la->area_shape);
  if (this->type != new_type) {
    shadow_discard_safe(shadows);
    this->type = new_type;
  }

  if (la->mode & LA_SHADOW) {
    if (la->type == LA_SUN) {
      if (this->shadow_id == LIGHT_NO_SHADOW) {
        this->shadow_id = shadows.directionals.alloc();
      }

      ShadowDirectional &shadow = shadows.directionals[this->shadow_id];
      shadow.sync(this->object_mat, la->bias * 0.05f, 1.0f);
    }
    else {
      float cone_aperture = DEG2RAD(360.0);
      if (la->type == LA_SPOT) {
        cone_aperture = min_ff(DEG2RAD(179.9), la->spotsize);
      }
      else if (la->type == LA_AREA) {
        cone_aperture = DEG2RAD(179.9);
      }

      if (this->shadow_id == LIGHT_NO_SHADOW) {
        this->shadow_id = shadows.punctuals.alloc();
      }

      ShadowPunctual &shadow = shadows.punctuals[this->shadow_id];
      shadow.sync(this->type,
                  this->object_mat,
                  cone_aperture,
                  la->clipsta,
                  this->influence_radius_max,
                  la->bias * 0.05f);
    }
  }
  else {
    shadow_discard_safe(shadows);
  }

  this->initialized = true;
}

void Light::shadow_discard_safe(ShadowModule &shadows)
{
  if (shadow_id != LIGHT_NO_SHADOW) {
    if (this->type != LIGHT_SUN) {
      shadows.punctuals.free(shadow_id);
    }
    else {
      shadows.directionals.free(shadow_id);
    }
    shadow_id = LIGHT_NO_SHADOW;
  }
}

/* Returns attenuation radius inversed & squared for easy bound checking inside the shader. */
float Light::attenuation_radius_get(const ::Light *la, float light_threshold, float light_power)
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
  return sqrtf(light_power / light_threshold);
}

void Light::shape_parameters_set(const ::Light *la, const float scale[3])
{
  if (la->type == LA_AREA) {
    float area_size_y = (ELEM(la->area_shape, LA_AREA_RECT, LA_AREA_ELLIPSE)) ? la->area_sizey :
                                                                                la->area_size;
    _area_size_x = max_ff(0.003f, la->area_size * scale[0] * 0.5f);
    _area_size_y = max_ff(0.003f, area_size_y * scale[1] * 0.5f);
    /* For volume point lighting. */
    radius_squared = max_ff(0.001f, hypotf(_area_size_x, _area_size_y) * 0.5f);
    radius_squared = square_f(radius_squared);
  }
  else {
    if (la->type == LA_SPOT) {
      /* Spot size & blend */
      spot_size_inv[0] = scale[2] / scale[0];
      spot_size_inv[1] = scale[2] / scale[1];
      float spot_size = cosf(la->spotsize * 0.5f);
      float spot_blend = (1.0f - spot_size) * la->spotblend;
      _spot_mul = 1.0f / max_ff(1e-8f, spot_blend);
      _spot_bias = -spot_size * _spot_mul;
    }

    if (la->type == LA_SUN) {
      _area_size_x = max_ff(0.001f, tanf(min_ff(la->sun_angle, DEG2RADF(179.9f)) / 2.0f));
      _area_size_y = _area_size_x;
    }
    else {
      _area_size_x = _area_size_y = max_ff(0.001f, la->area_size);
    }
    radius_squared = square_f(_area_size_x);
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
    power = 1.0f / (4.0f * square_f(_radius) * float(M_PI * M_PI));
  }
  else { /* LA_SUN */
    power = 1.0f / (square_f(_radius) * float(M_PI));
    /* Make illumination power closer to cycles for bigger radii. Cycles uses a cos^3 term that
     * we cannot reproduce so we account for that by scaling the light power. This function is
     * the result of a rough manual fitting. */
    /* Simplification of:
     * power *= 1 + r²/2 */
    power += 1.0f / (2.0f * M_PI);
  }
  return power;
}

float Light::point_power_get(const ::Light *la)
{
  /* Volume light is evaluated as point lights. Remove the shape power. */
  if (la->type == LA_AREA) {
    /* Match cycles. Empirical fit... must correspond to some constant. */
    float power = 0.0792f * M_PI;

    /* This corrects for area light most representative point trick. The fit was found by
     * reducing the average error compared to cycles. */
    float area = _area_size_x * _area_size_y;
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

void Light::debug_draw(void)
{
  const float color[4] = {0.8, 0.3, 0, 1};
  DRW_debug_sphere(_position, influence_radius_max, color);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name LightModule
 * \{ */

void LightModule::begin_sync(void)
{
  /* In begin_sync so it can be aninated. */
  float light_threshold = max_ff(1e-16f, inst_.scene->eevee.light_threshold);
  if (light_threshold != light_threshold_) {
    light_threshold_ = light_threshold;
    inst_.sampling.reset();
  }
}

void LightModule::sync_light(const Object *ob, ObjectHandle &handle)
{
  Light &light = lights_.lookup_or_add_default(handle.object_key);
  light.used = true;
  if (handle.recalc != 0 || !light.initialized) {
    light.sync(inst_.shadows, ob, light_threshold_);
  }
}

void LightModule::end_sync(void)
{
  lights_refs_.clear();

  Vector<ObjectKey, 0> deleted_keys;

  /* Detect light deletion. */
  for (auto item : lights_.items()) {
    Light &light = item.value;
    if (!light.used) {
      deleted_keys.append(item.key);
      light.shadow_discard_safe(inst_.shadows);
    }
    else {
      light.used = false;
      lights_refs_.append(&light);
    }
  }

  if (deleted_keys.size() > 0) {
    inst_.sampling.reset();
  }
  for (auto key : deleted_keys) {
    lights_.remove(key);
  }

  /* Call shadows.end_sync after light pruning to avoid packing deleted shadows. */
  inst_.shadows.end_sync();
}

/* Compute acceleration structure for the given view. If extent is 0, bind no lights. */
void LightModule::set_view(const DRWView *view, const ivec2 extent, bool enable_specular)
{
  if (extent.x == 0) {
    culling_.set_empty();
    return;
  }

  culling_.set_view(view, extent);

  for (auto light_id : lights_refs_.index_range()) {
    Light &light = *lights_refs_[light_id];

    BoundSphere bsphere;
    if (light.type == LIGHT_SUN) {
      /* Make sun lights cover the whole frustum. */
      float viewinv[4][4];
      DRW_view_viewmat_get(view, viewinv, true);
      copy_v3_v3(bsphere.center, viewinv[3]);
      bsphere.radius = fabsf(DRW_view_far_distance_get(view));
    }
    else {
      /* TODO(fclem) fit cones better. */
      copy_v3_v3(bsphere.center, light._position);
      bsphere.radius = light.influence_radius_max;
    }

    culling_.insert(light_id, bsphere);
  }

  DRW_view_set_active(view);

  /* This is only called if the light is visible under this view. */
  auto data_copy = [&](LightBatch &light_batch, uint32_t dst_index, uint32_t src_index) {
    Light &light = *this->lights_refs_[src_index];
    LightData &dst = light_batch.lights_data[dst_index];

    dst = light;
    if (!enable_specular) {
      dst.specular_power = 0.0f;
    }

    if (light.shadow_id != LIGHT_NO_SHADOW) {
      ShadowData &shadow_dst = light_batch.shadows_data[dst_index];
      if (light.type == LIGHT_SUN) {
        shadow_dst = this->inst_.shadows.directionals[light.shadow_id];
      }
      else {
        shadow_dst = this->inst_.shadows.punctuals[light.shadow_id];
      }
    }
  };

  /* Called for each batch. Do 2D gpu culling. */
  auto culling_func = [&](LightBatch &light_batch, CullingDataBuf &culling_data) {
    LightDataBuf &lights_data = light_batch.lights_data;
    ShadowDataBuf &shadows_data = light_batch.shadows_data;
    lights_data.push_update();
    shadows_data.push_update();

    this->inst_.shading_passes.light_culling.render(lights_data.ubo_get(), culling_data.ubo_get());
  };

  culling_.finalize(data_copy, culling_func);

  inst_.shadows.update_visible(view);
}

void LightModule::bind_batch(int batch_index)
{
  active_batch_ = batch_index;
  auto &batch = *culling_[batch_index];
  active_lights_ubo_ = batch.item_data.lights_data.ubo_get();
  active_shadows_ubo_ = batch.item_data.shadows_data.ubo_get();
  active_culling_ubo_ = batch.culling_ubo_get();
  active_culling_tx_ = batch.culling_texture_get();
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name CullingPass
 * \{ */

void CullingLightPass::sync(void)
{
  culling_ps_ = DRW_pass_create("CullingLight", DRW_STATE_WRITE_COLOR);

  GPUShader *sh = inst_.shaders.static_shader_get(CULLING_LIGHT);
  DRWShadingGroup *grp = DRW_shgroup_create(sh, culling_ps_);
  DRW_shgroup_uniform_block_ref(grp, "lights_block", &lights_ubo_);
  DRW_shgroup_uniform_block_ref(grp, "lights_culling_block", &culling_ubo_);
  DRW_shgroup_call_procedural_triangles(grp, nullptr, 1);
}

void CullingLightPass::render(const GPUUniformBuf *lights_ubo, const GPUUniformBuf *culling_ubo)
{
  lights_ubo_ = lights_ubo;
  culling_ubo_ = culling_ubo;
  DRW_draw_pass(culling_ps_);
}

/** \} */

}  // namespace blender::eevee
