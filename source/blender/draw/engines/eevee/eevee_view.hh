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
 */

#pragma once

#include "GPU_framebuffer.h"
#include "GPU_texture.h"

#include "DRW_render.h"

#include "eevee_renderpasses.hh"
#include "eevee_shaders.hh"

namespace blender::eevee {

typedef struct DeferredPass {
  DRWPass *test_ps_ = nullptr;
  ShaderModule *shaders_ = nullptr;

  void init(ShaderModule &shaders)
  {
    shaders_ = &shaders;

    DRWState state = DRW_STATE_WRITE_COLOR | DRW_STATE_WRITE_DEPTH | DRW_STATE_DEPTH_LESS;
    test_ps_ = DRW_pass_create("Deferred", state);
  }

  void surface_add(Object *ob, Material *mat, int matslot)
  {
    (void)mat;
    (void)matslot;
    GPUShader *sh = shaders_->static_shader_get(MESH);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, test_ps_);
    GPUBatch *geom = DRW_cache_object_surface_get(ob);
    DRW_shgroup_call(grp, geom, ob);
  }

  void render(void)
  {
    DRW_draw_pass(test_ps_);
  }
} DeferredPass;

typedef struct ShadingPasses {
  // BackgroundShadingPass background;
  DeferredPass opaque;

  void init(ShaderModule &shaders)
  {
    opaque.init(shaders);
  }
} ShadingPasses;

typedef struct ShadingView {
 private:
  /** Owned resources. */
  GPUFrameBuffer *view_fb_ = nullptr;
  /** Output render passes to accumulate to. */
  RenderPasses *render_passes_ = nullptr;
  /** Shading passes to render using this view. */
  ShadingPasses *shading_passes_ = nullptr;
  Camera *camera_ = nullptr;
  Sampling *sampling_ = nullptr;
  /** Draw resources. */
  GPUTexture *combined_tx_ = nullptr;
  GPUTexture *depth_tx_ = nullptr;
  /** View render target extent. */
  int extent_[2] = {-1, -1};
  /** View ID when rendering in panoramic projection mode. */
  int view_id_ = 0;

  const char *name_;

 public:
  ShadingView(void){};

  ~ShadingView()
  {
    GPU_framebuffer_free(view_fb_);
  }

  void configure(const char *name,
                 RenderPasses &render_passes,
                 ShadingPasses &shading_passes,
                 Sampling &sampling,
                 Camera &camera,
                 int view_id,
                 const int extent[2])
  {
    name_ = name;
    render_passes_ = &render_passes;
    shading_passes_ = &shading_passes;
    sampling_ = &sampling;
    camera_ = &camera;
    view_id_ = view_id;
    copy_v2_v2_int(extent_, extent);
  }

  void init(void)
  {
    /* HACK: View name should be unique and static.
     * With this, we can reuse the same texture across views. */
    DrawEngineType *owner = (DrawEngineType *)name_;

    // eRenderPassBit enabled_passes = render_passes_->enabled_passes_get();

    depth_tx_ = DRW_texture_pool_query_2d(UNPACK2(extent_), GPU_DEPTH24_STENCIL8, owner);
    combined_tx_ = DRW_texture_pool_query_2d(UNPACK2(extent_), GPU_RGBA16F, owner);

    GPU_framebuffer_ensure_config(&view_fb_,
                                  {
                                      GPU_ATTACHMENT_TEXTURE(depth_tx_),
                                      GPU_ATTACHMENT_TEXTURE(combined_tx_),
                                  });
  }

  void render(void)
  {
    float color[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    DRW_stats_group_start(name_);

    DRWView *drw_view = camera_->update_view(view_id_, extent_);
    DRW_view_set_active(drw_view);

    GPU_framebuffer_bind(view_fb_);
    GPU_framebuffer_clear_color_depth(view_fb_, color, 1.0f);
    shading_passes_->opaque.render();

    if (render_passes_->combined) {
      render_passes_->combined->accumulate(combined_tx_, drw_view);
    }

    if (render_passes_->depth) {
      render_passes_->depth->accumulate(depth_tx_, drw_view);
    }

    DRW_stats_group_end();
  }
} ShadingView;

}  // namespace blender::eevee