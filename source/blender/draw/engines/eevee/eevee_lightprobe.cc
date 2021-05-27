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

#include "eevee_instance.hh"

namespace blender::eevee {

void LightProbeModule::init()
{
  lightcache_ = static_cast<LightCache *>(inst_.scene->eevee.light_cache_data);

  // if (use_lookdev || lightcache_ == nullptr || lightcache_->validate() == false) {
  if (lookdev_lightcache_ == nullptr) {
    lookdev_lightcache_ = new LightCache();
  }
  lightcache_ = lookdev_lightcache_;
  // }
  // else {
  // OBJECT_GUARDED_SAFE_DELETE(lookdev_lightcache_, LightCache);
  // }

  for (DRWView *&view : face_view_) {
    view = nullptr;
  }
}

void LightProbeModule::begin_sync()
{
  cube_downsample_ps_ = DRW_pass_create("Downsample Cube", DRW_STATE_WRITE_COLOR);

  GPUShader *sh = inst_.shaders.static_shader_get(LIGHTPROBE_DOWNSAMPLE_CUBE);
  DRWShadingGroup *grp = DRW_shgroup_create(sh, cube_downsample_ps_);
  DRW_shgroup_uniform_texture_ref(grp, "input_tx", &cube_downsample_input_tx_);
  DRW_shgroup_call_procedural_triangles(grp, nullptr, 6);
}

void LightProbeModule::cubeface_winmat_get(mat4 &winmat, float near, float far)
{
  /* Simple 90° FOV projection. */
  perspective_m4(winmat, -near, near, -near, near, near, far);
}

void LightProbeModule::cubemap_prepare(vec3 &position, float near, float far)
{
  SceneEEVEE &sce_eevee = inst_.scene->eevee;
  int cube_res = sce_eevee.gi_cubemap_resolution;
  int cube_mip_count = (int)log2_ceil_u(cube_res);

  mat4 viewmat;
  unit_m4(viewmat);
  negate_v3_v3(viewmat[3], position);

  /* TODO(fclem) We might want to have theses as temporary textures. */
  cube_depth_tx_.ensure_cubemap("CubemapDepth", cube_res, cube_mip_count, GPU_DEPTH_COMPONENT32F);
  cube_color_tx_.ensure_cubemap("CubemapColor", cube_res, cube_mip_count, GPU_RGBA16F);

  cube_downsample_fb_.ensure(GPU_ATTACHMENT_TEXTURE(cube_depth_tx_),
                             GPU_ATTACHMENT_TEXTURE(cube_color_tx_));

  mat4 winmat;
  cubeface_winmat_get(winmat, near, far);
  for (int face : IndexRange(6)) {
    face_fb_[face].ensure(GPU_ATTACHMENT_TEXTURE_CUBEFACE_MIP(cube_depth_tx_, face, 0),
                          GPU_ATTACHMENT_TEXTURE_CUBEFACE_MIP(cube_color_tx_, face, 0));
    mat4 facemat;
    mul_m4_m4m4(facemat, winmat, cubeface_mat[face]);

    DRWView *&view = face_view_[face];
    if (view == nullptr) {
      view = DRW_view_create(viewmat, facemat, nullptr, nullptr, nullptr);
    }
    else {
      DRW_view_update(view, viewmat, facemat, nullptr, nullptr);
    }
  }
}

void LightProbeModule::update_world()
{
  vec3 position(0.0f);
  cubemap_prepare(position, 0.01f, 1.0f);

  auto probe_render = [&]() { inst_.shading_passes.background.render(); };
  cubemap_render(probe_render);

  // filter_glossy(0);

  /* TODO(fclem) Change ray type. */
  /* OPTI(fclem) Only re-render if there is a light path node in the world material. */
  // cubemap_render(probe_render);

  // filter_diffuse(0);

  // do_world_update = false;
}

void LightProbeModule::set_view(const DRWView *UNUSED(view), const ivec2 UNUSED(extent))
{
  if (do_world_update) {
    update_world();
  }
}

}  // namespace blender::eevee
