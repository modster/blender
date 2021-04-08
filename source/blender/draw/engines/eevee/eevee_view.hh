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

#include "eevee_depth_of_field.hh"
#include "eevee_renderpasses.hh"
#include "eevee_shader.hh"

namespace blender::eevee {

/* -------------------------------------------------------------------- */
/** \name Passes
 * \{ */

class DeferredPass {
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
};

class ShadingPasses {
 public:
  // BackgroundShadingPass background;
  DeferredPass opaque;

 public:
  ShadingPasses(ShaderModule &shaders) : opaque(shaders){};

  void sync()
  {
    opaque.sync();
  }
};

/** \} */

/* -------------------------------------------------------------------- */
/** \name ShadingView
 *
 * Render the scene and fill all render passes data.
 * \{ */

class ShadingView {
 private:
  Sampling &sampling_;
  /** Shading passes to render using this view. Shared with other views. */
  ShadingPasses &shading_passes_;
  /** Associated camera. */
  const Camera &camera_;
  /** Depth of field module. */
  DepthOfField dof_;

  /** Owned resources. */
  GPUFrameBuffer *view_fb_ = nullptr;
  /** Draw resources. Not owned. */
  GPUTexture *combined_tx_ = nullptr;
  GPUTexture *depth_tx_ = nullptr;
  GPUTexture *postfx_tx_ = nullptr;

  /** Main views is created from the camera (or is from the viewport). It is not jittered. */
  DRWView *main_view_ = nullptr;
  /** Sub views is jittered versions or the main views. This allows jitter updates without trashing
   * the visibility culling cache. */
  DRWView *sub_view_ = nullptr;
  /** Same as sub_view_ but has Depth Of Field jitter applied. */
  DRWView *render_view_ = nullptr;

  /** Render size of the view. Can change between scene sample eval. */
  int extent_[2] = {-1, -1};
  /** Static srting pointer. Used as debug name and as UUID for texture pool. */
  const char *name_;
  /** Matrix to apply to the viewmat. */
  const float (*face_matrix_)[4];

  bool is_enabled_ = false;

 public:
  ShadingView(ShaderModule &shaders,
              ShadingPasses &shading_passes,
              Sampling &sampling,
              const Camera &camera,
              const char *name,
              const float (*face_matrix)[4])
      : sampling_(sampling),
        shading_passes_(shading_passes),
        camera_(camera),
        dof_(shaders, sampling, name),
        name_(name),
        face_matrix_(face_matrix){};

  ~ShadingView()
  {
    GPU_framebuffer_free(view_fb_);
  }

  void init(const Scene *scene)
  {
    dof_.init(scene);
  }

  void sync(int render_extent_[2])
  {
    if (camera_.is_panoramic()) {
      int64_t render_pixel_count = render_extent_[0] * (int64_t)render_extent_[1];
      /* Divide pixel count between the 6 views. Rendering to a square target. */
      extent_[0] = extent_[1] = ceilf(sqrtf(1 + (render_pixel_count / 6)));
      /* TODO(fclem) Clip unused views heres. */
      is_enabled_ = true;
    }
    else {
      copy_v2_v2_int(extent_, render_extent_);
      /* Only enable -Z view. */
      is_enabled_ = (StringRefNull(name_) == "negZ_view");
    }

    if (!is_enabled_) {
      return;
    }

    /* Create views. */
    const CameraData &data = camera_.data_get();

    float viewmat[4][4], winmat[4][4];
    const float(*viewmat_p)[4] = viewmat, (*winmat_p)[4] = winmat;
    if (camera_.is_panoramic()) {
      /* TODO(fclem) Overscans. */
      float near = data.clip_near;
      float far = data.clip_far;
      perspective_m4(winmat, -near, near, -near, near, near, far);
      mul_m4_m4m4(viewmat, face_matrix_, data.viewmat);
    }
    else {
      viewmat_p = data.viewmat;
      winmat_p = data.winmat;
    }

    main_view_ = DRW_view_create(viewmat_p, winmat_p, nullptr, nullptr, nullptr);
    sub_view_ = DRW_view_create_sub(main_view_, viewmat_p, winmat_p);
    render_view_ = DRW_view_create_sub(main_view_, viewmat_p, winmat_p);

    dof_.sync(camera_, winmat_p, extent_);

    {
      /* Query temp textures and create framebuffers. */
      /* HACK: View name should be unique and static.
       * With this, we can reuse the same texture across views. */
      DrawEngineType *owner = (DrawEngineType *)name_;

      depth_tx_ = DRW_texture_pool_query_2d(UNPACK2(extent_), GPU_DEPTH24_STENCIL8, owner);
      combined_tx_ = DRW_texture_pool_query_2d(UNPACK2(extent_), GPU_RGBA16F, owner);
      /* TODO(fclem) Only allocate if needed. */
      postfx_tx_ = DRW_texture_pool_query_2d(UNPACK2(extent_), GPU_RGBA16F, owner);

      GPU_framebuffer_ensure_config(&view_fb_,
                                    {
                                        GPU_ATTACHMENT_TEXTURE(depth_tx_),
                                        GPU_ATTACHMENT_TEXTURE(combined_tx_),
                                    });
    }
  }

  void render(RenderPasses &render_passes)
  {
    if (!is_enabled_) {
      return;
    }

    float color[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    update_view();

    DRW_stats_group_start(name_);
    DRW_view_set_active(render_view_);

    GPU_framebuffer_bind(view_fb_);
    GPU_framebuffer_clear_color_depth(view_fb_, color, 1.0f);
    shading_passes_.opaque.render();

    GPUTexture *final_radiance_tx = render_post(combined_tx_);

    if (render_passes.combined) {
      render_passes.combined->accumulate(final_radiance_tx, sub_view_);
    }

    if (render_passes.depth) {
      render_passes.depth->accumulate(depth_tx_, sub_view_);
    }

    DRW_stats_group_end();
  }

  GPUTexture *render_post(GPUTexture *input_tx)
  {
    GPUTexture *output_tx = postfx_tx_;
    /* Swapping is done internally. Actual output is set to the next input. */
    dof_.render(depth_tx_, &input_tx, &output_tx);
    return input_tx;
  }

 private:
  void update_view(void)
  {
    float viewmat[4][4], winmat[4][4], persmat[4][4];
    DRW_view_viewmat_get(main_view_, viewmat, false);
    DRW_view_winmat_get(main_view_, winmat, false);
    DRW_view_persmat_get(main_view_, persmat, false);

    /* Anti-Aliasing / Super-Sampling jitter. */
    float jitter[2];
    sampling_.camera_aa_jitter_get(extent_, jitter);

    window_translate_m4(winmat, persmat, UNPACK2(jitter));
    DRW_view_update_sub(sub_view_, viewmat, winmat);

    /* FIXME(fclem): The offset may be is noticeably large and the culling might make object pop
     * out of the blurring radius. To fix this, use custom enlarged culling matrix. */
    dof_.jitter_apply(winmat, viewmat);
    DRW_view_update_sub(render_view_, viewmat, winmat);
  }
};

/** \} */

/* -------------------------------------------------------------------- */
/** \name Main View
 *
 * Container for all views needed to render the final image.
 * We might need up to 6 views for panoramic cameras.
 * \{ */

class MainView {
 private:
  std::array<ShadingView, 6> shading_views_;
  /** Internal render size. */
  int render_extent_[2];

 public:
  MainView(ShaderModule &shaders, ShadingPasses &shpasses, Camera &cam, Sampling &sampling)
      : shading_views_({
            ShadingView(shaders, shpasses, sampling, cam, "posX_view", cubeface_mat[0]),
            ShadingView(shaders, shpasses, sampling, cam, "negX_view", cubeface_mat[1]),
            ShadingView(shaders, shpasses, sampling, cam, "posY_view", cubeface_mat[2]),
            ShadingView(shaders, shpasses, sampling, cam, "negY_view", cubeface_mat[3]),
            ShadingView(shaders, shpasses, sampling, cam, "posZ_view", cubeface_mat[4]),
            ShadingView(shaders, shpasses, sampling, cam, "negZ_view", cubeface_mat[5]),
        })
  {
  }

  void init(const Scene *scene, const int full_extent_[2])
  {
    /* TODO(fclem) parameter hidden in experimental. We need to figure out mipmap bias to preserve
     * texture crispiness. */
    float resolution_scale = 1.0f;
    for (int i = 0; i < 2; i++) {
      render_extent_[i] = max_ii(1, roundf(full_extent_[i] * resolution_scale));
    }

    for (ShadingView &view : shading_views_) {
      view.init(scene);
    }
  }

  void sync(void)
  {
    for (ShadingView &view : shading_views_) {
      view.sync(render_extent_);
    }
  }

  void render(RenderPasses &render_passes)
  {
    for (ShadingView &view : shading_views_) {
      view.render(render_passes);
    }
  }
};

/** \} */

}  // namespace blender::eevee