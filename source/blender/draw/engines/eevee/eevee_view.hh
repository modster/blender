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

#pragma once

#include "GPU_framebuffer.h"
#include "GPU_texture.h"

#include "DRW_render.h"

#include "eevee_renderpasses.hh"
#include "eevee_shader.hh"

namespace blender::eevee {

/* -------------------------------------------------------------------- */
/** \name Passes
 * \{ */

typedef struct DeferredPass {
 private:
  ShaderModule &shaders_;

  DRWPass *test_ps_ = nullptr;

 public:
  DeferredPass(ShaderModule &shaders) : shaders_(shaders){};

  void sync()
  {
    DRWState state = DRW_STATE_WRITE_COLOR | DRW_STATE_WRITE_DEPTH | DRW_STATE_DEPTH_LESS;
    test_ps_ = DRW_pass_create("Deferred", state);
  }

  void surface_add(Object *ob, Material *mat, int matslot)
  {
    (void)mat;
    (void)matslot;
    GPUShader *sh = shaders_.static_shader_get(MESH);
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
 public:
  // BackgroundShadingPass background;
  DeferredPass opaque;

 public:
  ShadingPasses(ShaderModule &shaders) : opaque(shaders){};

  void sync()
  {
    opaque.sync();
  }
} ShadingPasses;

/** \} */

/* -------------------------------------------------------------------- */
/** \name ShadingView
 *
 * Render the scene and fill all render passes data.
 * \{ */

typedef struct ShadingView {
 private:
  /** Shading passes to render using this view. Shared with other views. */
  ShadingPasses &shading_passes_;
  /** Output render passes to accumulate to. */
  RenderPasses &render_passes_;
  /** Associated camera view. */
  const CameraView &camera_view_;

  /** Owned resources. */
  GPUFrameBuffer *view_fb_ = nullptr;
  /** Draw resources. Not owned. */
  GPUTexture *combined_tx_ = nullptr;
  GPUTexture *depth_tx_ = nullptr;

 public:
  ShadingView(RenderPasses &render_passes,
              ShadingPasses &shading_passes,
              const CameraView &camera_view)
      : shading_passes_(shading_passes),
        render_passes_(render_passes),
        camera_view_(camera_view){};

  ~ShadingView()
  {
    GPU_framebuffer_free(view_fb_);
  }

  void sync()
  {
    if (!camera_view_.is_enabled()) {
      return;
    }

    const int *extent = camera_view_.extent_get();

    /* HACK: View name should be unique and static.
     * With this, we can reuse the same texture across views. */
    DrawEngineType *owner = (DrawEngineType *)camera_view_.name_get();

    depth_tx_ = DRW_texture_pool_query_2d(UNPACK2(extent), GPU_DEPTH24_STENCIL8, owner);
    combined_tx_ = DRW_texture_pool_query_2d(UNPACK2(extent), GPU_RGBA16F, owner);

    GPU_framebuffer_ensure_config(&view_fb_,
                                  {
                                      GPU_ATTACHMENT_TEXTURE(depth_tx_),
                                      GPU_ATTACHMENT_TEXTURE(combined_tx_),
                                  });
  }

  void render(void)
  {
    if (!camera_view_.is_enabled()) {
      return;
    }

    float color[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    DRW_stats_group_start(camera_view_.name_get());

    const DRWView *drw_view = camera_view_.drw_view_get();
    DRW_view_set_active(drw_view);

    GPU_framebuffer_bind(view_fb_);
    GPU_framebuffer_clear_color_depth(view_fb_, color, 1.0f);
    shading_passes_.opaque.render();

    if (render_passes_.combined) {
      render_passes_.combined->accumulate(combined_tx_, drw_view);
    }

    if (render_passes_.depth) {
      render_passes_.depth->accumulate(depth_tx_, drw_view);
    }

    DRW_stats_group_end();
  }
} ShadingView;

/** \} */

/* -------------------------------------------------------------------- */
/** \name Main View
 *
 * Container for all views needed to render the final image.
 * We might need up to 6 views for panoramic cameras.
 * \{ */

typedef struct MainView {
 private:
  std::array<ShadingView, 6> shading_views_;

 public:
  MainView(RenderPasses &render_passes, ShadingPasses &shading_passes, const Camera &camera)
      : shading_views_({
            ShadingView(render_passes, shading_passes, camera.views_get()[0]),
            ShadingView(render_passes, shading_passes, camera.views_get()[1]),
            ShadingView(render_passes, shading_passes, camera.views_get()[2]),
            ShadingView(render_passes, shading_passes, camera.views_get()[3]),
            ShadingView(render_passes, shading_passes, camera.views_get()[4]),
            ShadingView(render_passes, shading_passes, camera.views_get()[5]),
        })
  {
  }

  void sync()
  {
    for (ShadingView &view : shading_views_) {
      view.sync();
    }
  }

  void render(void)
  {
    for (ShadingView &view : shading_views_) {
      view.render();
    }
  }
} MainView;

/** \} */

}  // namespace blender::eevee