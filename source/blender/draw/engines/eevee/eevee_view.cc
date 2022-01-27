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
 * A view is either:
 * - The entire main view.
 * - A fragment of the main view (for panoramic projections).
 * - A shadow map view.
 * - A lightprobe view (either planar, cubemap, irradiance grid).
 *
 * A pass is a container for scene data. It is view agnostic but has specific logic depending on
 * its type. Passes are shared between views.
 */

#include "BKE_global.h"
#include "DRW_render.h"

#include "eevee_instance.hh"

#include "eevee_view.hh"

namespace blender::eevee {

/* -------------------------------------------------------------------- */
/** \name ShadingView
 * \{ */

void ShadingView::init()
{
  dof_.init();
  mb_.init();
}

void ShadingView::sync(ivec2 render_extent_)
{
  if (inst_.camera.is_panoramic()) {
    int64_t render_pixel_count = render_extent_.x * (int64_t)render_extent_.y;
    /* Divide pixel count between the 6 views. Rendering to a square target. */
    extent_[0] = extent_[1] = ceilf(sqrtf(1 + (render_pixel_count / 6)));
    /* TODO(fclem) Clip unused views heres. */
    is_enabled_ = true;
  }
  else {
    extent_ = render_extent_;
    /* Only enable -Z view. */
    is_enabled_ = (StringRefNull(name_) == "negZ_view");
  }

  if (!is_enabled_) {
    return;
  }

  /* Create views. */
  const CameraData &data = inst_.camera.data_get();

  float viewmat[4][4], winmat[4][4];
  const float(*viewmat_p)[4] = viewmat, (*winmat_p)[4] = winmat;
  if (inst_.camera.is_panoramic()) {
    /* TODO(fclem) Overscans. */
    /* For now a mandatory 5% overscan for DoF. */
    float side = data.clip_near * 1.05f;
    float near = data.clip_near;
    float far = data.clip_far;
    perspective_m4(winmat, -side, side, -side, side, near, far);
    mul_m4_m4m4(viewmat, face_matrix_, data.viewmat);
  }
  else {
    viewmat_p = data.viewmat;
    winmat_p = data.winmat;
  }

  main_view_ = DRW_view_create(viewmat_p, winmat_p, nullptr, nullptr, nullptr);
  sub_view_ = DRW_view_create_sub(main_view_, viewmat_p, winmat_p);
  render_view_ = DRW_view_create_sub(main_view_, viewmat_p, winmat_p);

  dof_.sync(winmat_p, extent_);
  mb_.sync(extent_);
  velocity_.sync(extent_);
  rt_buffer_opaque_.sync(extent_);
  rt_buffer_refract_.sync(extent_);

  {
    /* Query temp textures and create framebuffers. */
    /* HACK: View name should be unique and static.
     * With this, we can reuse the same texture across views. */
    DrawEngineType *owner = (DrawEngineType *)name_;

    depth_tx_ = DRW_texture_pool_query_2d(UNPACK2(extent_), GPU_DEPTH24_STENCIL8, owner);
    combined_tx_ = DRW_texture_pool_query_2d(UNPACK2(extent_), GPU_RGBA16F, owner);
    /* TODO(fclem) Only allocate if needed. */
    postfx_tx_ = DRW_texture_pool_query_2d(UNPACK2(extent_), GPU_RGBA16F, owner);

    view_fb_.ensure(GPU_ATTACHMENT_TEXTURE(depth_tx_), GPU_ATTACHMENT_TEXTURE(combined_tx_));

    gbuffer_.sync(depth_tx_, combined_tx_, owner);
  }
}

void ShadingView::render(void)
{
  if (!is_enabled_) {
    return;
  }

  float color[4] = {0.0f, 0.0f, 0.0f, 1.0f};

  update_view();

  DRW_stats_group_start(name_);
  DRW_view_set_active(render_view_);

  GPU_framebuffer_bind(view_fb_);
  GPU_framebuffer_clear_color_depth(view_fb_, color, 1.0f);

  if (inst_.lookdev.render_background() == false) {
    inst_.shading_passes.background.render();
  }

  inst_.shading_passes.deferred.render(render_view_,
                                       gbuffer_,
                                       hiz_front_,
                                       hiz_back_,
                                       rt_buffer_opaque_,
                                       rt_buffer_refract_,
                                       view_fb_);

  inst_.lightprobes.draw_cache_display();

  inst_.lookdev.render_overlay(view_fb_);

  inst_.shading_passes.forward.render(render_view_, gbuffer_, hiz_front_, view_fb_);

  inst_.lights.debug_draw(view_fb_, hiz_front_);
  inst_.shadows.debug_draw(view_fb_, hiz_front_);

  velocity_.render(depth_tx_);

  if (inst_.render_passes.vector) {
    inst_.render_passes.vector->accumulate(velocity_.camera_vectors_get(), sub_view_);
  }

  GPUTexture *final_radiance_tx = render_post(combined_tx_);

  if (inst_.render_passes.combined) {
    inst_.render_passes.combined->accumulate(final_radiance_tx, sub_view_);
  }

  if (inst_.render_passes.depth) {
    inst_.render_passes.depth->accumulate(depth_tx_, sub_view_);
  }

  DRW_stats_group_end();
}

GPUTexture *ShadingView::render_post(GPUTexture *input_tx)
{
  GPUTexture *velocity_tx = velocity_.view_vectors_get();
  GPUTexture *output_tx = postfx_tx_;
  /* Swapping is done internally. Actual output is set to the next input. */
  dof_.render(depth_tx_, &input_tx, &output_tx);
  mb_.render(depth_tx_, velocity_tx, &input_tx, &output_tx);
  return input_tx;
}

void ShadingView::update_view(void)
{
  float viewmat[4][4], winmat[4][4];
  DRW_view_viewmat_get(main_view_, viewmat, false);
  DRW_view_winmat_get(main_view_, winmat, false);

  /* Anti-Aliasing / Super-Sampling jitter. */
  float jitter_u = 2.0f * (inst_.sampling.rng_get(SAMPLING_FILTER_U) - 0.5f) / extent_[0];
  float jitter_v = 2.0f * (inst_.sampling.rng_get(SAMPLING_FILTER_V) - 0.5f) / extent_[1];

  window_translate_m4(winmat, winmat, jitter_u, jitter_v);
  DRW_view_update_sub(sub_view_, viewmat, winmat);

  /* FIXME(fclem): The offset may be is noticeably large and the culling might make object pop
   * out of the blurring radius. To fix this, use custom enlarged culling matrix. */
  dof_.jitter_apply(winmat, viewmat);
  DRW_view_update_sub(render_view_, viewmat, winmat);

  inst_.lightprobes.set_view(render_view_, extent_);
  inst_.lights.set_view(render_view_, extent_);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name LightProbeView
 * \{ */

void LightProbeView::sync(Texture &color_tx,
                          Texture &depth_tx,
                          const mat4 winmat,
                          const mat4 viewmat,
                          bool is_only_background)
{
  mat4 facemat;
  mul_m4_m4m4(facemat, face_matrix_, viewmat);

  is_only_background_ = is_only_background;
  extent_ = ivec2(color_tx.width());
  view_ = DRW_view_create(facemat, winmat, nullptr, nullptr, nullptr);
  view_fb_.ensure(GPU_ATTACHMENT_TEXTURE_LAYER(depth_tx, layer_),
                  GPU_ATTACHMENT_TEXTURE_LAYER(color_tx, layer_));

  if (!is_only_background_) {
    /* Query temp textures and create framebuffers. */
    /* HACK: View name should be unique and static.
     * With this, we can reuse the same texture across views. */
    DrawEngineType *owner = (DrawEngineType *)name_;
    gbuffer_.sync(depth_tx, color_tx, owner, layer_);
    rt_buffer_opaque_.sync(extent_);
    rt_buffer_refract_.sync(extent_);
  }
}

void LightProbeView::render(void)
{
  if (!is_only_background_) {
    inst_.lightprobes.set_view(view_, extent_);
    inst_.lights.set_view(view_, extent_, false);
  }

  DRW_stats_group_start(name_);
  DRW_view_set_active(view_);

  GPU_framebuffer_bind(view_fb_);

  inst_.shading_passes.background.render();

  if (!is_only_background_) {
    GPU_framebuffer_clear_depth(view_fb_, 1.0f);

    inst_.shading_passes.deferred.render(
        view_, gbuffer_, hiz_front_, hiz_back_, rt_buffer_opaque_, rt_buffer_refract_, view_fb_);
    inst_.shading_passes.forward.render(view_, gbuffer_, hiz_front_, view_fb_);
  }
  DRW_stats_group_end();
}

/** \} */

}  // namespace blender::eevee