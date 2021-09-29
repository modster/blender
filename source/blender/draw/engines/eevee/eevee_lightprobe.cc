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

#include "BLI_rect.h"
#include "BLI_span.hh"
#include "DNA_defaults.h"
#include "DNA_lightprobe_types.h"

#include "eevee_instance.hh"

namespace blender::eevee {

void LightProbeModule::init()
{
  SceneEEVEE &sce_eevee = inst_.scene->eevee;

  lightcache_ = static_cast<LightCache *>(sce_eevee.light_cache_data);

  bool use_lookdev = inst_.use_studio_light();
  if (!use_lookdev && lightcache_ && lightcache_->load()) {
    OBJECT_GUARDED_SAFE_DELETE(lightcache_lookdev_, LightCache);
  }
  else {
    if (lightcache_ && (lightcache_->flag & LIGHTCACHE_NOT_USABLE)) {
      BLI_snprintf(
          inst_.info, sizeof(inst_.info), "Error: LightCache cannot be loaded on this GPU");
    }

    if (lightcache_lookdev_ == nullptr) {
      int cube_len = 1;
      int grid_len = 1;
      int irr_samples_len = 1;

      ivec3 irr_size;
      LightCache::irradiance_cache_size_get(
          sce_eevee.gi_visibility_resolution, irr_samples_len, irr_size);

      lightcache_lookdev_ = new LightCache(cube_len,
                                           grid_len,
                                           sce_eevee.gi_cubemap_resolution,
                                           sce_eevee.gi_visibility_resolution,
                                           irr_size);
    }
    lightcache_ = lightcache_lookdev_;
  }

  for (DRWView *&view : face_view_) {
    view = nullptr;
  }

  if (info_data_.cubes.display_size != sce_eevee.gi_cubemap_draw_size ||
      info_data_.grids.display_size != sce_eevee.gi_irradiance_draw_size ||
      info_data_.grids.irradiance_smooth != square_f(sce_eevee.gi_irradiance_smoothing)) {
    /* TODO(fclem) reset on scene update instead. */
    inst_.sampling.reset();
  }

  info_data_.cubes.display_size = sce_eevee.gi_cubemap_draw_size;
  info_data_.grids.display_size = sce_eevee.gi_irradiance_draw_size;
  info_data_.grids.irradiance_smooth = square_f(sce_eevee.gi_irradiance_smoothing);
  info_data_.grids.irradiance_cells_per_row = lightcache_->irradiance_cells_per_row_get();
  info_data_.grids.visibility_size = lightcache_->vis_res;
  info_data_.grids.visibility_cells_per_row = lightcache_->grid_tx.tex_size[0] /
                                              info_data_.grids.visibility_size;
  info_data_.grids.visibility_cells_per_layer = (lightcache_->grid_tx.tex_size[1] /
                                                 info_data_.grids.visibility_size) *
                                                info_data_.grids.visibility_cells_per_row;

  glossy_clamp_ = sce_eevee.gi_glossy_clamp;
  filter_quality_ = clamp_f(sce_eevee.gi_filter_quality, 1.0f, 8.0f);
}

void LightProbeModule::begin_sync()
{
  {
    cube_downsample_ps_ = DRW_pass_create("Downsample.Cube", DRW_STATE_WRITE_COLOR);

    GPUShader *sh = inst_.shaders.static_shader_get(LIGHTPROBE_FILTER_DOWNSAMPLE_CUBE);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, cube_downsample_ps_);
    DRW_shgroup_uniform_texture_ref(grp, "input_tx", &cube_downsample_input_tx_);
    DRW_shgroup_uniform_block(grp, "filter_block", filter_data_.ubo_get());
    DRW_shgroup_call_procedural_triangles(grp, nullptr, 6);
  }
  {
    filter_glossy_ps_ = DRW_pass_create("Filter.GlossyMip", DRW_STATE_WRITE_COLOR);

    GPUShader *sh = inst_.shaders.static_shader_get(LIGHTPROBE_FILTER_GLOSSY);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, filter_glossy_ps_);
    DRW_shgroup_uniform_texture_ref(grp, "radiance_tx", &cube_downsample_input_tx_);
    DRW_shgroup_uniform_block(grp, "filter_block", filter_data_.ubo_get());
    DRW_shgroup_call_procedural_triangles(grp, nullptr, 6);
  }
  {
    filter_diffuse_ps_ = DRW_pass_create("Filter.Diffuse", DRW_STATE_WRITE_COLOR);

    GPUShader *sh = inst_.shaders.static_shader_get(LIGHTPROBE_FILTER_DIFFUSE);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, filter_diffuse_ps_);
    DRW_shgroup_uniform_texture_ref(grp, "radiance_tx", &cube_downsample_input_tx_);
    DRW_shgroup_uniform_block(grp, "filter_block", filter_data_.ubo_get());
    DRW_shgroup_call_procedural_triangles(grp, nullptr, 1);
  }
  {
    filter_visibility_ps_ = DRW_pass_create("Filter.Visibility", DRW_STATE_WRITE_COLOR);

    GPUShader *sh = inst_.shaders.static_shader_get(LIGHTPROBE_FILTER_VISIBILITY);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, filter_visibility_ps_);
    DRW_shgroup_uniform_texture_ref(grp, "depth_tx", &cube_downsample_input_tx_);
    DRW_shgroup_uniform_block(grp, "filter_block", filter_data_.ubo_get());
    DRW_shgroup_call_procedural_triangles(grp, nullptr, 1);
  }

  display_ps_ = nullptr;

  if ((inst_.v3d != nullptr) && ((inst_.v3d->flag2 & V3D_HIDE_OVERLAYS) == 0)) {
    if (inst_.scene->eevee.flag & (SCE_EEVEE_SHOW_CUBEMAPS | SCE_EEVEE_SHOW_IRRADIANCE)) {
      DRWState state = DRW_STATE_WRITE_COLOR | DRW_STATE_WRITE_DEPTH | DRW_STATE_DEPTH_LESS;
      display_ps_ = DRW_pass_create("LightProbe.Display", state);
    }

    if (inst_.scene->eevee.flag & SCE_EEVEE_SHOW_CUBEMAPS) {
      if (lightcache_->cube_len > 1) {
        GPUShader *sh = inst_.shaders.static_shader_get(LIGHTPROBE_DISPLAY_CUBEMAP);
        DRWShadingGroup *grp = DRW_shgroup_create(sh, display_ps_);
        DRW_shgroup_uniform_texture_ref(grp, "lightprobe_cube_tx", cube_tx_ref_get());
        DRW_shgroup_uniform_block(grp, "cubes_block", cube_ubo_get());
        DRW_shgroup_uniform_block(grp, "lightprobes_info_block", info_ubo_get());

        uint cubemap_count = 0;
        /* Skip world. */
        for (auto cube_id : IndexRange(1, lightcache_->cube_len - 1)) {
          const LightProbeCache &cube = lightcache_->cube_data[cube_id];
          /* Note: only works because probes are rendered in sequential order. */
          if (cube.is_ready) {
            cubemap_count++;
          }
        }
        if (cubemap_count > 0) {
          DRW_shgroup_call_procedural_triangles(grp, nullptr, cubemap_count * 2);
        }
      }
    }

    if (inst_.scene->eevee.flag & SCE_EEVEE_SHOW_IRRADIANCE) {
      if (lightcache_->grid_len > 1) {
        GPUShader *sh = inst_.shaders.static_shader_get(LIGHTPROBE_DISPLAY_IRRADIANCE);
        DRWShadingGroup *grp = DRW_shgroup_create(sh, display_ps_);
        DRW_shgroup_uniform_texture_ref(grp, "lightprobe_grid_tx", grid_tx_ref_get());
        DRW_shgroup_uniform_block(grp, "grids_block", grid_ubo_get());
        DRW_shgroup_uniform_block(grp, "lightprobes_info_block", info_ubo_get());

        /* Skip world. */
        for (auto grid_id : IndexRange(1, lightcache_->grid_len - 1)) {
          const LightGridCache &grid = lightcache_->grid_data[grid_id];
          if (grid.is_ready) {
            DRWShadingGroup *grp_sub = DRW_shgroup_create_sub(grp);
            DRW_shgroup_uniform_int_copy(grp_sub, "grid_id", grid_id);
            uint sample_count = grid.resolution[0] * grid.resolution[1] * grid.resolution[2];
            DRW_shgroup_call_procedural_triangles(grp_sub, nullptr, sample_count * 2);
          }
        }
      }
    }
  }
}

void LightProbeModule::end_sync()
{
  if (lightcache_->flag & LIGHTCACHE_UPDATE_WORLD) {
    cubemap_prepare(vec3(0.0f), 0.01f, 1.0f, true);
  }
}

void LightProbeModule::cubeface_winmat_get(mat4 &winmat, float near, float far)
{
  /* Simple 90° FOV projection. */
  perspective_m4(winmat, -near, near, -near, near, near, far);
}

void LightProbeModule::cubemap_prepare(vec3 position, float near, float far, bool background_only)
{
  SceneEEVEE &sce_eevee = inst_.scene->eevee;
  int cube_res = sce_eevee.gi_cubemap_resolution;
  int cube_mip_count = (int)log2_ceil_u(cube_res);

  mat4 viewmat;
  unit_m4(viewmat);
  negate_v3_v3(viewmat[3], position);

  /* TODO(fclem) We might want to have theses as temporary textures. */
  cube_depth_tx_.ensure_cubemap("CubemapDepth", cube_res, cube_mip_count, GPU_DEPTH24_STENCIL8);
  cube_color_tx_.ensure_cubemap("CubemapColor", cube_res, cube_mip_count, GPU_RGBA16F);
  GPU_texture_mipmap_mode(cube_color_tx_, true, true);

  cube_downsample_fb_.ensure(GPU_ATTACHMENT_TEXTURE(cube_depth_tx_),
                             GPU_ATTACHMENT_TEXTURE(cube_color_tx_));

  filter_cube_fb_.ensure(GPU_ATTACHMENT_NONE, GPU_ATTACHMENT_TEXTURE(lightcache_->cube_tx.tex));
  filter_grid_fb_.ensure(GPU_ATTACHMENT_NONE, GPU_ATTACHMENT_TEXTURE(lightcache_->grid_tx.tex));

  mat4 winmat;
  cubeface_winmat_get(winmat, near, far);

  for (auto i : IndexRange(ARRAY_SIZE(probe_views_))) {
    probe_views_[i].sync(cube_color_tx_, cube_depth_tx_, winmat, viewmat, background_only);
  }
}

void LightProbeModule::cubemap_render(void)
{
  DRW_stats_group_start("Cubemap Render");
  for (auto i : IndexRange(ARRAY_SIZE(probe_views_))) {
    probe_views_[i].render();
  }
  DRW_stats_group_end();

  /* Update mipchain. */
  filter_data_.target_layer = 0;
  filter_data_.push_update();
  cube_downsample_input_tx_ = cube_color_tx_;

  DRW_stats_group_start("Cubemap Downsample");
  GPU_framebuffer_recursive_downsample(cube_downsample_fb_, 7, cube_downsample_cb, this);
  DRW_stats_group_end();
}

void LightProbeModule::filter_glossy(int cube_index, float intensity)
{
  DRW_stats_group_start("Filter.Glossy");

  filter_data_.instensity_fac = intensity;
  filter_data_.target_layer = cube_index * 6;

  int level_max = lightcache_->mips_len;
  for (int level = 0; level <= level_max; level++) {
    filter_data_.luma_max = (glossy_clamp_ > 0.0f) ? glossy_clamp_ : 1e16f;
    /* Disney Roughness. */
    filter_data_.roughness = square_f(level / (float)level_max);
    /* Distribute Roughness across lod more evenly. */
    filter_data_.roughness = square_f(filter_data_.roughness);
    /* Avoid artifacts. */
    filter_data_.roughness = clamp_f(filter_data_.roughness, 1e-4f, 0.9999f);
    /* Variable sample count and bias to make first levels faster. */
    switch (level) {
      case 0:
        filter_data_.sample_count = 1.0f;
        filter_data_.lod_bias = -1.0f;
        break;
      case 1:
        filter_data_.sample_count = filter_quality_ * 32.0f;
        filter_data_.lod_bias = 1.0f;
        break;
      case 2:
        filter_data_.sample_count = filter_quality_ * 40.0f;
        filter_data_.lod_bias = 2.0f;
        break;
      case 3:
        filter_data_.sample_count = filter_quality_ * 64.0f;
        filter_data_.lod_bias = 2.0f;
        break;
      default:
        filter_data_.sample_count = filter_quality_ * 128.0f;
        filter_data_.lod_bias = 2.0f;
        break;
    }
    /* Add automatic LOD bias (based on target size). */
    filter_data_.lod_bias += lod_bias_from_cubemap();

    filter_data_.push_update();

    filter_cube_fb_.ensure(GPU_ATTACHMENT_NONE,
                           GPU_ATTACHMENT_TEXTURE_MIP(lightcache_->cube_tx.tex, level));
    GPU_framebuffer_bind(filter_cube_fb_);
    DRW_draw_pass(filter_glossy_ps_);
  }

  DRW_stats_group_end();
}

void LightProbeModule::filter_diffuse(int sample_index, float intensity)
{
  filter_data_.instensity_fac = intensity;
  filter_data_.target_layer = 0;
  filter_data_.luma_max = 1e16f;
  filter_data_.sample_count = 1024.0f;
  filter_data_.lod_bias = lod_bias_from_cubemap();

  filter_data_.push_update();

  ivec2 extent = ivec2(3, 2);
  ivec2 offset = extent;
  offset.x *= sample_index % info_data_.grids.irradiance_cells_per_row;
  offset.y *= sample_index / info_data_.grids.irradiance_cells_per_row;

  GPU_framebuffer_bind(filter_grid_fb_);
  GPU_framebuffer_viewport_set(filter_grid_fb_, UNPACK2(offset), UNPACK2(extent));
  DRW_draw_pass(filter_diffuse_ps_);
  GPU_framebuffer_viewport_reset(filter_grid_fb_);
}

void LightProbeModule::filter_visibility(int sample_index,
                                         float visibility_blur,
                                         float visibility_range)
{
  ivec2 extent = ivec2(info_data_.grids.visibility_size);
  ivec2 offset = extent;
  offset.x *= sample_index % info_data_.grids.visibility_cells_per_row;
  offset.y *= (sample_index / info_data_.grids.visibility_cells_per_row) %
              info_data_.grids.visibility_cells_per_layer;

  filter_data_.target_layer = 1 + sample_index / info_data_.grids.visibility_cells_per_layer;
  filter_data_.sample_count = 512.0f; /* TODO refine */
  filter_data_.visibility_blur = visibility_blur;
  filter_data_.visibility_range = visibility_range;

  filter_data_.push_update();

  GPU_framebuffer_bind(filter_grid_fb_);
  GPU_framebuffer_viewport_set(filter_grid_fb_, UNPACK2(offset), UNPACK2(extent));
  DRW_draw_pass(filter_visibility_ps_);
  GPU_framebuffer_viewport_reset(filter_grid_fb_);
}

void LightProbeModule::update_world_cache()
{
  DRW_stats_group_start("LightProbe.world");

  const DRWView *view_active = DRW_view_get_active();

  cubemap_render();

  filter_diffuse(0, 1.0f);

  if ((lightcache_->flag & LIGHTCACHE_NO_REFLECTION) == 0) {
    /* TODO(fclem) Change ray type. */
    /* OPTI(fclem) Only re-render if there is a light path node in the world material. */
    // cubemap_render();

    filter_glossy(0, 1.0f);
  }

  if (view_active != nullptr) {
    DRW_view_set_active(view_active);
  }

  DRW_stats_group_end();
}

/* Ensure a temporary cache the same size at the target lightcache exists. */
LightCache *LightProbeModule::baking_cache_get(void)
{
  if (lightcache_baking_ == nullptr) {
    lightcache_baking_ = new LightCache(lightcache_->cube_len,
                                        lightcache_->grid_len,
                                        lightcache_->cube_tx.tex_size[0],
                                        lightcache_->vis_res,
                                        lightcache_->grid_tx.tex_size);

    if (lightcache_baking_->flag != LIGHTCACHE_INVALID) {
      LightCache &lcache_src = *lightcache_;
      LightCache &lcache = *lightcache_baking_;
      /* Copy cache structure. */
      memcpy(lcache.cube_data, lcache_src.cube_data, lcache.cube_len * sizeof(*lcache.cube_data));
      memcpy(lcache.grid_data, lcache_src.grid_data, lcache.grid_len * sizeof(*lcache.grid_data));

      /* Make grids renderable. */
      for (LightGridCache &grid : MutableSpan(lcache.grid_data, lcache.grid_len)) {
        grid.is_ready = 1;
      }
      /* Avoid sampling further than mip 0. Mips > 0 being undefined. */
      lcache.mips_len = 0;
      lcache.flag |= LIGHTCACHE_NO_REFLECTION;

      /* Init to black. */
      uint data_cube = 0;
      uchar data_grid[4] = {0, 0, 0, 0};
      GPU_texture_clear(lcache.cube_tx.tex, GPU_DATA_10_11_11_REV, &data_cube);
      GPU_texture_clear(lcache.grid_tx.tex, GPU_DATA_UBYTE, &data_grid);
    }
  }
  return lightcache_baking_;
}

void LightProbeModule::bake(Depsgraph *depsgraph,
                            int type,
                            int index,
                            int bounce,
                            const float position[3],
                            const LightProbe *probe,
                            float visibility_range)
{
  rcti rect;
  BLI_rcti_init(&rect, 0, 0, 1, 1);

  /* Disable screenspace effects. */
  SceneEEVEE &sce_eevee = DEG_get_evaluated_scene(depsgraph)->eevee;
  sce_eevee.flag &= ~(SCE_EEVEE_GTAO_ENABLED | SCE_EEVEE_SSR_ENABLED);

  inst_.init(ivec2(1), &rect, nullptr, depsgraph, probe);
  inst_.sampling.reset();
  inst_.render_sync();
  inst_.sampling.step();

  float near = (probe) ? probe->clipsta : 0.1f;
  float far = (probe) ? probe->clipend : 1.0f;
  float intensity = (probe) ? probe->intensity : 1.0f;

  bool background_only = (probe == nullptr);
  cubemap_prepare(position, near, far, background_only);

  if (type == LIGHTPROBE_TYPE_CUBE && probe != nullptr) {
    /* Reflections cubemaps are rendered after all irradiance bounces.
     * Swap to get the final irradiance in lightcache_baking_. */
    swap_irradiance_cache();
  }

  /* Render using the previous bounce to light the scene. */
  lightcache_ = baking_cache_get();

  cubemap_render();

  /* Filter on the original cache. */
  lightcache_ = reinterpret_cast<LightCache *>(sce_eevee.light_cache_data);

  if (type == LIGHTPROBE_TYPE_CUBE) {
    filter_glossy(index, intensity);
    /* Swap back final irradiance to lightcache_. */
    if (probe != nullptr) {
      swap_irradiance_cache();
    }
  }
  else {
    filter_diffuse(index, intensity);
    if (probe && bounce < 2) {
      /* No need to filter visibility after 2nd bounce since both lightcache_ and
       * lightcache_baking_ will have correct visibility grid. */
      filter_visibility(index, probe->vis_blur, visibility_range);
    }
  }
}

/* Push world probe to first grid and cubemap slots. */
void LightProbeModule::sync_world(const DRWView *view)
{
  BoundSphere view_bounds = DRW_view_frustum_bsphere_get(view);
  /* Playing safe. The fake grid needs to be bigger than the frustum. */
  view_bounds.radius = clamp_f(view_bounds.radius * 2.0, 0.0f, FLT_MAX);

  CubemapData &cube = cube_data_[0];
  GridData &grid = grid_data_[0];

  scale_m4_fl(grid.local_mat, view_bounds.radius);
  negate_v3_v3(grid.local_mat[3], view_bounds.center);
  copy_m4_m4(cube.influence_mat, grid.local_mat);
  copy_m4_m4(cube.parallax_mat, cube.influence_mat);

  grid.resolution = ivec3(1);
  grid.offset = 0;
  grid.level_skip = 1;
  grid.attenuation_bias = 0.001f;
  grid.attenuation_scale = 1.0f;
  grid.visibility_range = 1.0f;
  grid.visibility_bleed = 0.001f;
  grid.visibility_bias = 0.0f;
  grid.increment_x = vec3(0.0f);
  grid.increment_y = vec3(0.0f);
  grid.increment_z = vec3(0.0f);
  grid.corner = vec3(0.0f);

  cube._parallax_type = CUBEMAP_SHAPE_SPHERE;
  cube._layer = 0.0;
}

void LightProbeModule::sync_grid(const DRWView *UNUSED(view),
                                 const LightGridCache &grid_cache,
                                 int grid_index)
{
  /* Skip the world probe. */
  if (grid_index == 0 || grid_cache.is_ready != 1) {
    return;
  }
  GridData &grid = grid_data_[info_data_.grids.grid_count];
  copy_m4_m4(grid.local_mat, grid_cache.mat);
  grid.resolution = ivec3(grid_cache.resolution);
  grid.offset = grid_cache.offset;
  grid.level_skip = grid_cache.level_bias;
  grid.attenuation_bias = grid_cache.attenuation_bias;
  grid.attenuation_scale = grid_cache.attenuation_scale;
  grid.visibility_range = grid_cache.visibility_range;
  grid.visibility_bleed = grid_cache.visibility_bleed;
  grid.visibility_bias = grid_cache.visibility_bias;
  grid.increment_x = vec3(grid_cache.increment_x);
  grid.increment_y = vec3(grid_cache.increment_y);
  grid.increment_z = vec3(grid_cache.increment_z);
  grid.corner = vec3(grid_cache.corner);

  info_data_.grids.grid_count++;
}

void LightProbeModule::sync_cubemap(const DRWView *UNUSED(view),
                                    const LightProbeCache &cube_cache,
                                    int cube_index)
{
  /* Skip the world probe. */
  if (cube_index == 0 || cube_cache.is_ready != 1) {
    return;
  }
  CubemapData &cube = cube_data_[info_data_.cubes.cube_count];
  copy_m4_m4(cube.parallax_mat, cube_cache.parallaxmat);
  copy_m4_m4(cube.influence_mat, cube_cache.attenuationmat);
  cube._attenuation_factor = cube_cache.attenuation_fac;
  cube._attenuation_type = cube_cache.attenuation_type;
  cube._parallax_type = cube_cache.parallax_type;
  cube._layer = cube_index;
  cube._world_position_x = cube_cache.position[0];
  cube._world_position_y = cube_cache.position[1];
  cube._world_position_z = cube_cache.position[2];

  info_data_.cubes.cube_count++;
}

/* Only enables world light probe if extent is invalid (no culling possible). */
void LightProbeModule::set_view(const DRWView *view, const ivec2 extent)
{
  if (lightcache_->flag & LIGHTCACHE_UPDATE_WORLD) {
    /* Set before update to avoid infinite recursion. */
    lightcache_->flag &= ~LIGHTCACHE_UPDATE_WORLD;
    update_world_cache();
  }

  /* Only sync when setting the view. This way we can cull probes not in frustum. */
  /* TODO(fclem) implement culling. But needs to fix display when not all probes are present. */
  info_data_.grids.grid_count = 1;
  info_data_.cubes.cube_count = 1;

  sync_world(view);
  /* Only world if extent is 0. */
  if (extent.x > 0) {
    for (auto i : IndexRange(lightcache_->grid_len)) {
      sync_grid(view, lightcache_->grid_data[i], i);
    }
    for (auto i : IndexRange(lightcache_->cube_len)) {
      sync_cubemap(view, lightcache_->cube_data[i], i);
    }
  }

  info_data_.cubes.roughness_max_lod = lightcache_->mips_len;
  inst_.lookdev.rotation_get(info_data_.cubes.lookdev_rotation);
  inst_.lookdev.rotation_get(info_data_.grids.lookdev_rotation);

  active_grid_tx_ = lightcache_->grid_tx.tex;
  active_cube_tx_ = lightcache_->cube_tx.tex;

  info_data_.push_update();
  grid_data_.push_update();
  cube_data_.push_update();
}

void LightProbeModule::draw_cache_display(void)
{
  /* Only draws something if enabled. */
  DRW_draw_pass(display_ps_);
}

}  // namespace blender::eevee
