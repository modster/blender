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

namespace blender::eevee {

/* -------------------------------------------------------------------- */
/** \name Shadow Module
 *
 * \{ */

void ShadowModule::begin_sync(void)
{
  regions_.clear();
}

/* Return first shadow region index or -1 if could not allocate shadow region. */
int ShadowModule::sync_punctual_shadow(const ObjectHandle &ob_handle,
                                       const mat4 light_mat,
                                       float radius,
                                       float cone_aperture,
                                       float near_clip)
{
  ShadowPunctual &shadow = point_shadows_.lookup_or_add_default(ob_handle.object_key);

  if (shadow.region_count == -1 || ob_handle.recalc != 0) {
    shadow.do_update = true;
    shadow.sync(light_mat, radius, cone_aperture, near_clip, 64);
  }
  shadow.alive = true;

  shadow.region_first = regions_.size();
  shadow.allocate_regions(regions_);
  shadow.region_count = regions_.size() - shadow.region_first;

  if (regions_.size() > SHADOW_REGION_MAX) {
    shadow.region_first = -1;
  }
  return shadow.region_first;
}

void ShadowModule::end_sync(void)
{
  atlas_tx_.ensure(4096, 4096, 1, GPU_DEPTH_COMPONENT16);
  atlas_fb_.ensure(GPU_ATTACHMENT_TEXTURE(atlas_tx_));
  atlas_tx_ptr_ = atlas_tx_;
  GPU_texture_compare_mode(atlas_tx_, true);
  GPU_texture_filter_mode(atlas_tx_, true);

  /* Sort regions by size. */
  /* TODO */

  /* Pack regions. */
  /* Simple packing for now because every region has the same width. */
  ivec2 offset = {0, 0};
  for (ShadowRegion &region : regions_) {
    if (offset[1] + region.extent[1] > atlas_tx_.height()) {
      offset[0] += region.extent[0];
      offset[1] = 0;
    }
    copy_v2_v2_int(region.offset, offset);
    offset[1] += region.extent[1];

    /* Init view pointer to avoid reusing after free. */
    region.view = nullptr;
  }
}

/* Update all shadow regions visible inside the view. */
void ShadowModule::set_view(const DRWView *view)
{
  /* Create DRWViews if needed. */
  for (ShadowPunctual &shpoint : point_shadows_.values()) {
    if (soft_shadows_enabled_) {
      shpoint.do_update = true;
    }
    if (shpoint.region_first == -1 || !shpoint.do_update ||
        !DRW_culling_sphere_test(view, &shpoint.bsphere)) {
      continue;
    }
    MutableSpan<ShadowRegion> sh_regions = regions_.as_mutable_span().slice(shpoint.region_first,
                                                                            shpoint.region_count);
    shpoint.prepare_views(sh_regions);
    shpoint.do_update = false;

    for (ShadowRegion &region : sh_regions) {
      region.do_update = true;
    }
  }

  DRW_stats_group_start("ShadowUpdate");

  int atlas_extent[2] = {atlas_tx_.width(), atlas_tx_.height()};
  /* Render all updated views.
   * This is because culling will be faster on multiple views at once. */
  for (auto i : regions_.index_range()) {
    ShadowRegion &region = regions_[i];
    if (!region.do_update) {
      continue;
    }
    region.do_update = false;

    DRW_view_set_active(region.view);

    GPU_framebuffer_bind(atlas_fb_);
    GPU_framebuffer_viewport_set(atlas_fb_, UNPACK2(region.offset), UNPACK2(region.extent));

    inst_.shading_passes.shadow.render();

    regions_data_[i] = region.to_region_data(atlas_extent);
  }

  regions_data_.push_update();

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
     * - Dedicated Image View & framebuffer for each ShadowRegion.
     */
    DRWState state = DRW_STATE_WRITE_DEPTH | DRW_STATE_DEPTH_ALWAYS | DRW_STATE_CULL_BACK;
    clear_ps_ = DRW_pass_create("ShadowClear", state);

    GPUShader *sh = inst_.shaders.static_shader_get(SHADOW_CLEAR);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, clear_ps_);
    DRW_shgroup_call_procedural_triangles(grp, nullptr, 1);
  }
  {
    DRWState state = DRW_STATE_WRITE_DEPTH | DRW_STATE_DEPTH_LESS | DRW_STATE_SHADOW_OFFSET |
                     DRW_STATE_CULL_BACK;
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
