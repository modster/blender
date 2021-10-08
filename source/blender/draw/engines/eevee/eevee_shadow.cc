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
 * The shadow module manages shadow update tagging & shadow rendering.
 */

#include "eevee_instance.hh"
#include <iostream>

namespace blender::eevee {

/* -------------------------------------------------------------------- */
/** \name Shadow Punctual
 *
 * \{ */

void ShadowPunctual::sync(eLightType light_type,
                          const mat4 &object_mat,
                          float cone_aperture,
                          float near_clip,
                          float far_clip,
                          float bias)
{
  bool is_wide_cone = cone_aperture > DEG2RAD(90);
  bool is_omni = cone_aperture > DEG2RAD(180);
  if ((is_wide_cone_ != is_wide_cone) || (is_omni_ != is_omni)) {
    is_wide_cone_ = is_wide_cone;
    is_omni_ = is_omni;
    shadows_.packing_changed_ = true;
  }

  far_ = max_ff(far_clip, 3e-4f);
  near_ = min_ff(near_clip, far_clip - 1e-4f);
  bias_ = bias;

  update_extent(cube_res_);

  light_type_ = light_type;

  /* Keep custom data. */
  size_x_ = _area_size_x;
  size_y_ = _area_size_y;

  copy_m4_m4(object_mat_.values, object_mat);
  /* Clear custom data. */
  object_mat_.values[0][3] = object_mat_.values[1][3] = object_mat_.values[2][3] = 0.0f;
  object_mat_.values[3][3] = 1.0f;

  /* TODO(fclem) Tighter bounds for cones. */
  copy_v3_v3(bsphere.center, object_mat_.translation());
  bsphere.radius = far_clip;

  do_update_tag = true;
}

void ShadowPunctual::update_extent(int cube_res)
{
  do_update_persist = true;
  is_visible = false;
  cube_res_ = cube_res;

  if (is_omni_) {
    extent[0] = cube_res_;
    extent[1] = cube_res_ * 6;
  }
  else if (is_wide_cone_) {
    extent[0] = cube_res_;
    extent[1] = cube_res_ * 3;
  }
  else {
    extent[0] = cube_res_;
    extent[1] = cube_res_;
  }
}

void ShadowPunctual::render(Instance &inst, MutableSpan<DRWView *> views)
{
  mat4 viewmat;
  copy_m4_m4(viewmat, object_mat_.values);

  if (inst.shadows.soft_shadows_enabled_) {
    *reinterpret_cast<vec3 *>(viewmat[3]) += object_mat_.ref_3x3() * random_offset_;
  }

  /* The viewmat is garanteed to be normalized by Light::Light(), so transpose is also the
   * inverse. Transpose is equivalent to inverse only if matrix is symmetrical. */
  /* TODO(fclem) */
  // transpose_m4(r_viewmat);
  /* Apply translation. */
  // translate_m4(r_viewmat, );
  invert_m4(viewmat);
  /* Create/Update all views before rendering. This is in order to perform culling in batch. */
  mat4 winmat;
  cubeface_winmat_get(winmat, false);
  view_update(views[Z_NEG], viewmat, winmat, Z_NEG);
  if (is_wide_cone_) {
    if (!is_omni_) {
      cubeface_winmat_get(winmat, true);
    }
    view_update(views[X_POS], viewmat, winmat, X_POS);
    view_update(views[X_NEG], viewmat, winmat, X_NEG);
    view_update(views[Y_POS], viewmat, winmat, Y_POS);
    view_update(views[Y_NEG], viewmat, winmat, Y_NEG);
  }
  if (is_omni_) {
    view_update(views[Z_POS], viewmat, winmat, Z_POS);
  }

  /* TODO(fclem): Optimization: only update faces that are intersecting the active view. */

  view_render(inst, views[Z_NEG], Z_NEG);
  if (is_wide_cone_) {
    view_render(inst, views[X_POS], X_POS);
    view_render(inst, views[X_NEG], X_NEG);
    view_render(inst, views[Y_POS], Y_POS);
    view_render(inst, views[Y_NEG], Y_NEG);
  }
  if (is_omni_) {
    view_render(inst, views[Z_POS], Z_POS);
  }

  do_update_persist = false;
  is_visible = false;
}

/* Sets random offset for soft shadows. This needs to be called before rendering and before
 * converting the data to ShadowPunctualData, whichever comes first. If it is not the case,
 * rendering and shadow map sampling won't match. */
void ShadowPunctual::random_position_on_shape_set(Instance &inst)
{
  if (inst.shadows.soft_shadows_enabled_ == false) {
    random_offset_ = vec3(0.0f);
    aa_offset_ = vec2(0.0f);
    return;
  }

  float random[3];
  random[0] = inst.sampling.rng_get(SAMPLING_SHADOW_U);
  random[1] = inst.sampling.rng_get(SAMPLING_SHADOW_V);
  random[2] = inst.sampling.rng_get(SAMPLING_SHADOW_W);

  aa_offset_.x = inst.sampling.rng_get(SAMPLING_SHADOW_X);
  aa_offset_.y = inst.sampling.rng_get(SAMPLING_SHADOW_Y);

  /* TODO(fclem) Decorellate per shadow. */

  switch (light_type_) {
    case LIGHT_RECT: {
      random_offset_ = vec3((random[0] > 0.5f ? random[0] - 1.0f : random[0]) * 2.0f * size_x_,
                            (random[1] > 0.5f ? random[1] - 1.0f : random[1]) * 2.0f * size_y_,
                            0.0f);
      break;
    }
    case LIGHT_ELLIPSE: {
      vec2 disk = inst.sampling.sample_disk(random);
      random_offset_ = vec3(disk.x * size_x_, disk.y * size_y_, 0.0f);
      break;
    }
    default: {
      random_offset_ = inst.sampling.sample_ball(random) * size_x_;
      break;
    }
  }
}

void ShadowPunctual::cubeface_winmat_get(mat4 &winmat, bool half_opened)
{
  /**
   *             o < Shadow source
   *            / \
   *           /   \
   *          /     \ < Face projection border
   *         /       \
   *        /         \
   *       /           \
   * | x | x | x | x | x | x |
   *
   * Here x's denote the samples location and cube_res_ is 6.
   * To match samples at the border of each projector, we need a half pixel offset on each side of
   * the projection. We add 2 more pixels to allow for AA jittering. Note that jittering this
   * matrix directly exhibit the seam at face borders if the samples don't line up with the
   * face projection border.
   **/
  float padding = 1.5f * 2.0f;
  float side = near_ * (cube_res_ + padding) / cube_res_;
  vec2 aa_jitter = aa_offset_ * (side * 2.0f / cube_res_);
  perspective_m4(winmat,
                 -side + aa_jitter.x,
                 side + aa_jitter.x,
                 -side + aa_jitter.y,
                 ((half_opened) ? 0.0f : side) + aa_jitter.y,
                 near_,
                 far_);
}

void ShadowPunctual::view_update(DRWView *&view,
                                 const mat4 &viewmat,
                                 const mat4 &winmat,
                                 eShadowCubeFace face)
{
  mat4 facemat;
  mul_m4_m4m4(facemat, shadow_face_mat[face], viewmat);

  if (view == nullptr) {
    view = DRW_view_create(facemat, winmat, nullptr, nullptr, nullptr);
  }
  else {
    DRW_view_update(view, facemat, winmat, nullptr, nullptr);
  }
}

void ShadowPunctual::view_render(Instance &inst, DRWView *view, eShadowCubeFace face)
{
  AtlasRegion sub_region = *this;
  sub_region.extent[0] = sub_region.extent[1] = cube_res_;
  if (face > 0 && !is_omni_) {
    sub_region.extent[1] /= 2;
    sub_region.offset[1] += sub_region.extent[1] * (face + 1);
  }
  else {
    sub_region.offset[1] += sub_region.extent[1] * face;
  }
  DRW_view_set_active(view);
  GPU_viewport(UNPACK2(sub_region.offset), UNPACK2(sub_region.extent));
  inst.shading_passes.shadow.render();
}

ShadowPunctual::operator ShadowPunctualData()
{
  ShadowPunctualData data;
  cubeface_winmat_get(data.shadow_mat, false);
  /**
   * Conversion from NDC to atlas coordinate.
   * GLSL pseudo code:
   * region_uv = ((ndc * 0.5 + 0.5) * cube_res + offset) / atlas_extent;
   * region_uv = ndc * (0.5 * cube_res) / atlas_extent
   *                 + (0.5 * cube_res + offset) / atlas_extent;
   * Also remove half a pixel from each side to avoid interpolation issues.
   * The projection matrix should take this into account.
   **/
  mat4 coord_mat;
  zero_m4(coord_mat);
  coord_mat[0][0] = (0.5f * cube_res_) / shadows_.atlas_extent_[0];
  coord_mat[1][1] = (0.5f * cube_res_) / shadows_.atlas_extent_[1];
  coord_mat[2][2] = 0.5f;
  coord_mat[3][0] = (0.5f * cube_res_ + offset[0]) / shadows_.atlas_extent_[0];
  coord_mat[3][1] = (0.5f * cube_res_ + offset[1]) / shadows_.atlas_extent_[1];
  coord_mat[3][2] = 0.5f;
  coord_mat[3][3] = 1.0f;
  mul_m4_m4m4(data.shadow_mat, coord_mat, data.shadow_mat);

  data.shadow_offset = random_offset_;
  data.is_omni = is_omni_;
  data.shadow_bias = bias_;
  data.region_offset = (is_omni_ ? cube_res_ : (cube_res_ / 2)) / float(shadows_.atlas_extent_.y);
  data.shadow_near = near_;
  data.shadow_far = far_;
  return data;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Shadow Module
 *
 * \{ */

void ShadowModule::init(void)
{
  if (cube_shadow_res_ != inst_.scene->eevee.shadow_cube_size) {
    cube_shadow_res_ = inst_.scene->eevee.shadow_cube_size;
    for (ShadowPunctual &shadow : punctuals_) {
      shadow.update_extent(cube_shadow_res_);
    }
    packing_changed_ = true;
    inst_.sampling.reset();
  }

  eGPUTextureFormat shadow_format = (inst_.scene->eevee.flag & SCE_EEVEE_SHADOW_HIGH_BITDEPTH) ?
                                        GPU_DEPTH_COMPONENT32F :
                                        GPU_DEPTH_COMPONENT16;
  if (shadow_format_ != shadow_format) {
    shadow_format_ = shadow_format;
    format_changed_ = true;
    inst_.sampling.reset();
  }

  const bool soft_shadow_enabled = (inst_.scene->eevee.flag & SCE_EEVEE_SHADOW_SOFT) != 0;
  if (soft_shadows_enabled_ != soft_shadow_enabled) {
    soft_shadows_enabled_ = soft_shadow_enabled;
    for (ShadowPunctual &shadow : punctuals_) {
      shadow.do_update_persist = true;
    }
    inst_.sampling.reset();
  }

  memset(views_, 0, sizeof(views_));
}

void ShadowModule::sync_caster(Object *ob, const ObjectHandle &handle)
{
  ShadowCaster &caster = casters_.lookup_or_add_default(handle.object_key);
  caster.used = true;
  if (handle.recalc != 0 || !caster.initialized) {
    caster.sync(ob);
  }
}

/* Packs all shadow regions into a shadow atlas and update shadow casters linking / updates. */
void ShadowModule::end_sync(void)
{
  const bool do_global_update = packing_changed_ || format_changed_;

  Vector<ObjectKey, 0> deleted_keys;

  /* Collect updated shadows bits. */
  punctuals_used_bits_.resize(punctuals_.size());
  punctuals_updated_bits_.resize(punctuals_.size());
  for (auto shadow_index : punctuals_.index_range()) {
    ShadowPunctual &shpoint = punctuals_[shadow_index];
    punctuals_used_bits_.set_bit(shadow_index, shpoint.used);
    punctuals_updated_bits_.set_bit(shadow_index, shpoint.do_update_tag);
    shpoint.do_update_tag = false;
  }

  /* Search for deleted shadow casters or if shadow caster WAS in shadow radius. */
  for (auto item : casters_.items()) {
    ShadowCaster &caster = item.value;
    if (!caster.used) {
      deleted_keys.append(item.key);
      caster.updated = true;
    }

    if (!do_global_update && caster.updated) {
      /* If the shadow-caster has been deleted or updated, update previously intersect shadows. */
      /* TODO(fclem): Bitmap iterator. */
      for (auto shadow_index : caster.intersected_shadows_bits.index_range()) {
        if (caster.intersected_shadows_bits[shadow_index] == false) {
          continue;
        }
        punctuals_[shadow_index].do_update_persist = true;
      }
    }

    if (caster.used) {
      /* If the shadow-caster has been updated, update all shadow intersection bits and
       * tag the shadows for update. If the shadow-caster has NOT been updated, only update shadow
       * intersection bits for updated shadows. */
      ShadowBitmap &shadow_bits = (caster.updated) ? punctuals_used_bits_ :
                                                     punctuals_updated_bits_;

      caster.intersected_shadows_bits.resize(punctuals_.size());
      /* If the shadow-caster has been updated, update intersected shadows and their bits. */
      /* TODO(fclem): This part can be slow (max O(N²), min O(N)), optimize it with an acceleration
       * structure. */
      /* TODO(fclem): Bitmap iterator. */
      for (auto shadow_index : shadow_bits.index_range()) {
        if (shadow_bits[shadow_index] == false) {
          continue;
        }
        ShadowPunctual &shadow = punctuals_[shadow_index];
        const bool isect = ShadowCaster::intersect(caster, shadow);
        caster.intersected_shadows_bits.set_bit(shadow_index, isect);
        if (isect) {
          punctuals_[shadow_index].do_update_persist = true;
        }
      }

      /* TODO(fclem) Gather global bounds. */
    }
    caster.updated = false;
    caster.used = false;
  }

  if (deleted_keys.size() > 0) {
    inst_.sampling.reset();
  }
  for (auto key : deleted_keys) {
    casters_.remove(key);
  }

  if (!do_global_update) {
    return;
  }

  /* Pruned unused shadows at the end of the vector. */
  while (punctuals_.size() > 0 && punctuals_.last().used == false) {
    punctuals_.remove_last();
    punctual_unused_count_--;
  }
  if (punctual_unused_first_ >= punctuals_.size()) {
    /* First unused has been pruned. */
    punctual_unused_first_ = INT_MAX;
  }

  Vector<AtlasRegion *> regions;
  /* Collect regions. */
  for (auto i : punctuals_.index_range()) {
    ShadowPunctual &shpoint = punctuals_[i];
    if (shpoint.used) {
      /* TODO(fclem): Ideally, we could de-fragment the atlas using compute passes to move regions.
       * For now, we update all shadows inside the atlas. */
      shpoint.do_update_persist = true;
      shpoint.is_visible = false;
      regions.append(&shpoint);
    }
  }

  /* Sort regions by size. */
  /* TODO */

  /* Pack regions. */
  /* TODO(fclem) better packing. */
  int row_height = cube_shadow_res_ * 6;
  const int max_width = 8192;
  ivec2 offset = {0, 0};
  for (AtlasRegion *region : regions) {
    if (offset[0] + region->extent[0] > max_width) {
      offset[1] += row_height;
      offset[0] = 0;
    }
    region->offset = offset;
    offset[0] += region->extent[0];
  }
  ivec2 extent = {(offset.y > 0) ? max_width : offset.x,
                  (offset.y > 0) ? (offset.y + row_height) : row_height};

  /* TODO(fclem) Make allocation safe. */
  atlas_tx_.ensure(UNPACK2(extent), 1, shadow_format_);
  atlas_fb_.ensure(GPU_ATTACHMENT_TEXTURE(atlas_tx_));
  atlas_tx_ptr_ = atlas_tx_;
  GPU_texture_compare_mode(atlas_tx_, true);
  GPU_texture_filter_mode(atlas_tx_, true);

  atlas_extent_ = {atlas_tx_.width(), atlas_tx_.height()};

  format_changed_ = false;
  packing_changed_ = false;
}

/* Update all shadow regions visible inside the view. */
void ShadowModule::update_visible(const DRWView *UNUSED(view))
{
  bool force_update = false;
  if (soft_shadows_enabled_ && (inst_.sampling.sample_get() != last_sample_)) {
    force_update = true;
    last_sample_ = inst_.sampling.sample_get();
  }
  else {
    last_sample_ = 0;
  }

  DRW_stats_group_start("ShadowUpdate");

  GPU_framebuffer_bind(atlas_fb_);

  for (ShadowPunctual &shpoint : punctuals_) {
    if (force_update) {
      shpoint.do_update_persist = true;
    }
    if (shpoint.used && shpoint.do_update_persist && shpoint.is_visible) {
      shpoint.render(inst_, MutableSpan(views_, 6));
    }
  }

  DRW_stats_group_end();
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Shadow Pass
 *
 * \{ */

void ShadowPass::sync(void)
{
  {
    /* Clear using a fullscreen quad.
     * This is to avoid using scissors & to make it Metal compatible.
     * Other possible way to do it:
     * - Compute shader.
     * - Blit blank region.
     * - Render to tmp buffer. Copy to atlas later.
     * - Dedicated Image View & framebuffer for each AtlasRegion.
     */
    DRWState state = DRW_STATE_WRITE_DEPTH | DRW_STATE_DEPTH_ALWAYS;
    clear_ps_ = DRW_pass_create("ShadowClear", state);

    GPUShader *sh = inst_.shaders.static_shader_get(SHADOW_CLEAR);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, clear_ps_);
    DRW_shgroup_call_procedural_triangles(grp, nullptr, 1);
  }
  {
    DRWState state = DRW_STATE_WRITE_DEPTH | DRW_STATE_DEPTH_LESS | DRW_STATE_SHADOW_OFFSET;
    surface_ps_ = DRW_pass_create("ShadowSurface", state);
  }
}

DRWShadingGroup *ShadowPass::material_add(::Material *UNUSED(material), GPUMaterial *gpumat)
{
  DRWShadingGroup *grp = DRW_shgroup_material_create(gpumat, surface_ps_);
  DRW_shgroup_uniform_block(grp, "sampling_block", inst_.sampling.ubo_get());
  return grp;
}

void ShadowPass::render(void)
{
  DRW_draw_pass(clear_ps_);
  DRW_draw_pass(surface_ps_);
}

/** \} */

}  // namespace blender::eevee
