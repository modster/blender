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

typedef struct EEVEE_DeferredPass {
  DRWPass *test_ps_ = nullptr;
  EEVEE_Shaders *shaders_ = nullptr;

  void init(EEVEE_Shaders &shaders)
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
} EEVEE_DeferredPass;

typedef struct EEVEE_ShadingPasses {
  // EEVEE_BackgroundShadingPass background;
  EEVEE_DeferredPass opaque;

  void init(EEVEE_Shaders &shaders)
  {
    opaque.init(shaders);
  }
} EEVEE_ShadingPasses;

typedef struct EEVEE_ShadingView {
 private:
  /** Owned resources. */
  GPUFrameBuffer *view_fb_ = nullptr;
  /** Output render passes to accumulate to. */
  EEVEE_RenderPasses *render_passes_ = nullptr;
  /** Shading passes to render using this view. */
  EEVEE_ShadingPasses *shading_passes_ = nullptr;
  EEVEE_Camera *camera_ = nullptr;
  EEVEE_Random *random_ = nullptr;
  /** Draw resources. */
  GPUTexture *combined_tx_ = nullptr;
  GPUTexture *depth_tx_ = nullptr;
  DRWView *drw_view_ = nullptr;
  /** View render target resolution. */
  int res_[2] = {-1, -1};

  const char *name_;

 public:
  EEVEE_ShadingView(void){};

  ~EEVEE_ShadingView(void)
  {
    GPU_framebuffer_free(view_fb_);
  }

  void configure(const char *name,
                 EEVEE_RenderPasses &render_passes,
                 EEVEE_ShadingPasses &shading_passes,
                 EEVEE_Random &random,
                 EEVEE_Camera &camera,
                 const int resolution[2])
  {
    name_ = name;
    render_passes_ = &render_passes;
    shading_passes_ = &shading_passes;
    random_ = &random;
    camera_ = &camera;
    copy_v2_v2_int(res_, resolution);
  }

  void init(void)
  {
    /* HACK: View name should be unique and static.
     * With this, we can reuse the same texture across views. */
    DrawEngineType *owner = (DrawEngineType *)name_;

    // eEEVEERenderPassBit enabled_passes = render_passes_->enabled_passes_get();

    depth_tx_ = DRW_texture_pool_query_2d(UNPACK2(res_), GPU_DEPTH24_STENCIL8, owner);
    combined_tx_ = DRW_texture_pool_query_2d(UNPACK2(res_), GPU_RGBA16F, owner);

    GPU_framebuffer_ensure_config(&view_fb_,
                                  {
                                      GPU_ATTACHMENT_TEXTURE(depth_tx_),
                                      GPU_ATTACHMENT_TEXTURE(combined_tx_),
                                  });

    // drw_view_ = DRW_view_create();
  }

  void render(void)
  {
    float color[4] = {0.5f, 1.0f, 0.1f, 1.0f};
    // DRW_view_update(drw_view_, );

    GPU_framebuffer_bind(view_fb_);
    GPU_framebuffer_clear_color_depth(view_fb_, color, 1.0f);
    shading_passes_->opaque.render();

    if (render_passes_->combined) {
      render_passes_->combined->accumulate(combined_tx_, drw_view_);
    }

    if (render_passes_->depth) {
      render_passes_->depth->accumulate(depth_tx_, drw_view_);
    }
  }
} EEVEE_ShadingView;
