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

void ShadowPunctual::sync(
    const mat4 &object_mat, float cone_aperture, float near_clip, float far_clip, float bias)
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

  copy_m4_m4(object_mat_, object_mat);
  /* Clear custom data. */
  object_mat_[0][3] = object_mat_[1][3] = object_mat_[2][3] = 0.0f;
  object_mat_[3][3] = 1.0f;
}

void ShadowPunctual::update_extent(int cube_res)
{
  do_update = true;
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
  /* TODO(fclem) jitter. */
  copy_m4_m4(viewmat, object_mat_);

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

  do_update = false;
}

void ShadowPunctual::cubeface_winmat_get(mat4 &winmat, bool half_opened)
{
  /* Open the frustum a bit more to align border pixels with the different views. */
  float side = near_ * (cube_res_ / float(cube_res_ - 1));
  perspective_m4(winmat, -side, side, -side, (half_opened) ? 0.0f : side, near_, far_);
}

void ShadowPunctual::view_update(DRWView *&view,
                                 const mat4 &viewmat,
                                 const mat4 &winmat,
                                 eShadowCubeFace face)
{
  mat4 facemat;
  mul_m4_m4m4(facemat, winmat, shadow_face_mat[face]);

  if (view == nullptr) {
    view = DRW_view_create(viewmat, facemat, nullptr, nullptr, nullptr);
  }
  else {
    DRW_view_update(view, viewmat, facemat, nullptr, nullptr);
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

  data.is_omni = is_omni_;
  data.shadow_bias = bias_;
  data.region_offset = (is_omni_ ? cube_res_ : (cube_res_ / 2)) / float(shadows_.atlas_extent_.y);
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

  soft_shadows_enabled_ = (inst_.scene->eevee.flag & SCE_EEVEE_SHADOW_SOFT) != 0;
}

/* Packs all shadow regions into a shadow atlas. */
void ShadowModule::end_sync(void)
{
  if (!packing_changed_ && !format_changed_) {
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
      shpoint.do_update = true;
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
  DRW_stats_group_start("ShadowUpdate");

  GPU_framebuffer_bind(atlas_fb_);

  for (ShadowPunctual &shpoint : punctuals_) {
    if (shpoint.used && (shpoint.do_update || soft_shadows_enabled_) && shpoint.is_visible) {
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

    GPUShader *sh = inst_.shaders.static_shader_get(DEPTH_SIMPLE_MESH);
    surface_grp_ = DRW_shgroup_create(sh, surface_ps_);
  }
}

void ShadowPass::surface_add(Object *ob, Material *mat, int matslot)
{
  (void)mat;
  (void)matslot;
  GPUBatch *geom = DRW_cache_object_surface_get(ob);
  if (geom == nullptr) {
    return;
  }

  DRW_shgroup_call(surface_grp_, geom, ob);
}

void ShadowPass::render(void)
{
  DRW_draw_pass(clear_ps_);
  DRW_draw_pass(surface_ps_);
}

/** \} */

}  // namespace blender::eevee
