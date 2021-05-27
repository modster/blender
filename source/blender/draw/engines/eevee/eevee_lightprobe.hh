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

#include "eevee_wrapper.hh"

namespace blender::eevee {

class Instance;

class LightProbeModule {
 private:
  Instance &inst_;

  /* Either scene lightcache or lookdev lightcache */
  LightCache *lightcache_ = nullptr;
  /* Own lightcache used for lookdev lighting or as fallback. */
  LightCache *lookdev_lightcache_ = nullptr;

  /* Used for rendering probes. */
  /* OPTI(fclem) Share for the whole scene? Only allocate temporary? */
  Texture cube_depth_tx_ = Texture("CubemapDepth");
  Texture cube_color_tx_ = Texture("CubemapColor");
  std::array<Framebuffer, 6> face_fb_ = {Framebuffer("posX_view"),
                                         Framebuffer("negX_view"),
                                         Framebuffer("posY_view"),
                                         Framebuffer("negY_view"),
                                         Framebuffer("posZ_view"),
                                         Framebuffer("negZ_view")};

  Framebuffer cube_downsample_fb_ = Framebuffer("CubeDownsample");

  std::array<DRWView *, 6> face_view_ = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};

  DRWPass *cube_downsample_ps_ = nullptr;
  DRWPass *filter_glossy_ps_ = nullptr;
  DRWPass *filter_diffuse_ps_ = nullptr;

  bool do_world_update = false;
  /** Input texture to downsample cube pass. */
  GPUTexture *cube_downsample_input_tx_ = nullptr;

 public:
  LightProbeModule(Instance &inst) : inst_(inst){};
  ~LightProbeModule()
  {
    OBJECT_GUARDED_SAFE_DELETE(lookdev_lightcache_, LightCache);
  }

  void init();

  void begin_sync();

  void set_view(const DRWView *view, const ivec2 extent);

  void set_world_dirty(void)
  {
    do_world_update = true;
  }

 private:
  void update_world();

  void cubeface_winmat_get(mat4 &winmat, float near, float far);

  void cubemap_prepare(vec3 &position, float near, float far);

  static void cube_downsample_cb(void *thunk, int UNUSED(level))
  {
    DRW_draw_pass(reinterpret_cast<LightProbeModule *>(thunk)->cube_downsample_ps_);
  }

  template<typename RenderF> void cubemap_render(const RenderF &render_callback)
  {
    for (auto face_id : IndexRange(6)) {
      DRW_view_set_active(face_view_[face_id]);
      GPU_framebuffer_bind(face_fb_[face_id]);
      render_callback();
    }
    /* Update all mipmaps. */
    cube_downsample_input_tx_ = cube_color_tx_;
    GPU_framebuffer_recursive_downsample(cube_downsample_fb_, 8, cube_downsample_cb, this);
  }
};

}  // namespace blender::eevee
