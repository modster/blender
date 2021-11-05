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
 * Copyright 2018, Blender Foundation.
 */

/** \file
 * \ingroup eevee
 */

#pragma once

#include "eevee_lightcache.hh"
#include "eevee_view.hh"

#include "eevee_wrapper.hh"

namespace blender::eevee {

class Instance;

class LightProbeModule {
 private:
  Instance &inst_;

  LightProbeFilterDataBuf filter_data_;
  LightProbeInfoDataBuf info_data_;
  GridDataBuf grid_data_;
  CubemapDataBuf cube_data_;

  /* Either scene lightcache or lookdev lightcache */
  LightCache *lightcache_ = nullptr;
  /* Own lightcache used for lookdev lighting or as fallback. */
  LightCache *lightcache_lookdev_ = nullptr;
  /* Temporary cache used for baking. */
  LightCache *lightcache_baking_ = nullptr;

  /* Used for rendering probes. */
  /* OPTI(fclem) Share for the whole scene? Only allocate temporary? */
  Texture cube_depth_tx_ = Texture("CubemapDepth");
  Texture cube_color_tx_ = Texture("CubemapColor");
  LightProbeView probe_views_[6];

  Framebuffer cube_downsample_fb_ = Framebuffer("cube_downsample");
  Framebuffer filter_cube_fb_ = Framebuffer("filter_cube");
  Framebuffer filter_grid_fb_ = Framebuffer("filter_grid");

  std::array<DRWView *, 6> face_view_ = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};

  DRWPass *cube_downsample_ps_ = nullptr;
  DRWPass *filter_glossy_ps_ = nullptr;
  DRWPass *filter_diffuse_ps_ = nullptr;
  DRWPass *filter_visibility_ps_ = nullptr;

  DRWPass *display_ps_ = nullptr;

  /** Input texture to downsample cube pass. */
  GPUTexture *cube_downsample_input_tx_ = nullptr;
  /** Copy of actual textures from the lightcache_. */
  GPUTexture *active_grid_tx_ = nullptr;
  GPUTexture *active_cube_tx_ = nullptr;
  /** Constant values during baking. */
  float glossy_clamp_ = 0.0;
  float filter_quality_ = 0.0;

 public:
  LightProbeModule(Instance &inst)
      : inst_(inst),
        probe_views_{{inst, "posX_view", cubeface_mat[0], 0},
                     {inst, "negX_view", cubeface_mat[1], 1},
                     {inst, "posY_view", cubeface_mat[2], 2},
                     {inst, "negY_view", cubeface_mat[3], 3},
                     {inst, "posZ_view", cubeface_mat[4], 4},
                     {inst, "negZ_view", cubeface_mat[5], 5}}
  {
  }

  ~LightProbeModule()
  {
    OBJECT_GUARDED_SAFE_DELETE(lightcache_lookdev_, LightCache);
    OBJECT_GUARDED_SAFE_DELETE(lightcache_baking_, LightCache);
  }

  void init();

  void begin_sync();
  void end_sync();

  void set_view(const DRWView *view, const ivec2 extent);

  void set_world_dirty(void)
  {
    lightcache_->flag |= LIGHTCACHE_UPDATE_WORLD;
  }

  void swap_irradiance_cache(void)
  {
    if (lightcache_baking_ && lightcache_) {
      SWAP(GPUTexture *, lightcache_baking_->grid_tx.tex, lightcache_->grid_tx.tex);
    }
  }

  const GPUUniformBuf *grid_ubo_get() const
  {
    return grid_data_.ubo_get();
  }
  const GPUUniformBuf *cube_ubo_get() const
  {
    return cube_data_.ubo_get();
  }
  const GPUUniformBuf *info_ubo_get() const
  {
    return info_data_.ubo_get();
  }
  GPUTexture **grid_tx_ref_get()
  {
    return &active_grid_tx_;
  }
  GPUTexture **cube_tx_ref_get()
  {
    return &active_cube_tx_;
  }

  void bake(Depsgraph *depsgraph,
            int type,
            int index,
            int bounce,
            const float position[3],
            const LightProbe *probe = nullptr,
            float visibility_range = 0.0f);

  void draw_cache_display(void);

 private:
  void update_world_cache();

  void sync_world(const DRWView *view);
  void sync_grid(const DRWView *view, const struct LightGridCache &grid_cache, int grid_index);
  void sync_cubemap(const DRWView *view, const struct LightProbeCache &cube_cache, int cube_index);

  LightCache *baking_cache_get(void);

  void cubemap_prepare(vec3 position, float near, float far, bool background_only);

  void filter_glossy(int cube_index, float intensity);
  void filter_diffuse(int sample_index, float intensity);
  void filter_visibility(int sample_index, float visibility_blur, float visibility_range);

  float lod_bias_from_cubemap(void)
  {
    float target_size_sq = square_f(GPU_texture_width(cube_color_tx_));
    return 0.5f * logf(target_size_sq / filter_data_.sample_count) / logf(2);
  }

  static void cube_downsample_cb(void *thunk, int UNUSED(level))
  {
    DRW_draw_pass(reinterpret_cast<LightProbeModule *>(thunk)->cube_downsample_ps_);
  }

  void cubemap_render(void);
};

}  // namespace blender::eevee
