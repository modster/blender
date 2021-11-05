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

#include "BKE_global.h"

#include "eevee_instance.hh"

#include "draw_manager_text.h"

#include <iostream>

namespace blender::eevee {

/* -------------------------------------------------------------------- */
/** \name Tile map
 *
 * \{ */

void ShadowTileMap::sync_clipmap(const Camera &camera,
                                 const float4x4 &object_mat_,
                                 const AABB &local_casters_bounds,
                                 int clipmap_level)
{
  if (is_cubeface || (level != clipmap_level) ||
      float3(object_mat.values[2]) != float3(object_mat_.values[2])) {
    /* Direction or clipping changed. */
    set_dirty();
  }
  is_cubeface = false;
  level = clipmap_level;
  near = -local_casters_bounds.max.z + 1e-8f;
  far = -local_casters_bounds.min.z - 1e-8f;

  object_mat = object_mat_;
  /* Clear embedded custom data. */
  object_mat.values[0][3] = object_mat.values[1][3] = object_mat.values[2][3] = 0.0f;
  object_mat.values[3][3] = 1.0f;
  /* Remove translation. */
  zero_v3(object_mat.values[3]);

  /* Same as object_mat.inverted() because object_mat it is orthogonal. */
  float4x4 viewmat = object_mat.transposed();
  float3 tilemap_center = viewmat * camera.position();

  /* In world units. */
  /* NOTE(fclem): If we would to introduce a global scaling option it would be here. */
  float map_size = tilemap_coverage_get();
  float tile_size = map_size / tile_map_resolution;
  /* TODO(fclem): Add hysteresis in viewport to avoid too much invalidation. */
  /* Snap to tile of the grid above. This avoid half visible tiles and too much update. */
  float upper_tile_size = tile_size * 2.0f;
  /* Snap to a position on tile sized grid. */
  int2 new_offset;
  new_offset.x = roundf(tilemap_center.x / upper_tile_size) * upper_tile_size;
  new_offset.y = roundf(tilemap_center.y / upper_tile_size) * upper_tile_size;
  grid_shift = new_offset - grid_offset;
  grid_offset = new_offset;

  tilemap_center = object_mat * float3(grid_offset.x, grid_offset.y, 0.0f);

  copy_v3_v3(object_mat.values[3], tilemap_center);

  float half_size = tilemap_coverage_get() / 2.0f;
  float4x4 winmat;
  orthographic_m4(winmat.values, -half_size, half_size, -half_size, half_size, near, far);
  viewmat = object_mat.inverted_affine();
  mul_m4_m4m4(persmat, winmat.values, viewmat.values);
}

void ShadowTileMap::sync_cubeface(
    const float4x4 &object_mat_, const float4x4 &winmat, float near_, float far_, eCubeFace face)
{
  if (!is_cubeface || (cubeface != face) || (near != near_) || (far != far_) ||
      float3(object_mat.values[2]) != float3(object_mat_.values[2])) {
    /* Direction or clipping changed. */
    set_dirty();
  }
  is_cubeface = true;
  cubeface = face;
  near = near_;
  far = far_;

  object_mat = object_mat_;
  /* Clear embedded custom data. */
  object_mat.values[0][3] = object_mat.values[1][3] = object_mat.values[2][3] = 0.0f;
  object_mat.values[3][3] = 1.0f;

  float4x4 viewmat = float4x4(shadow_face_mat[cubeface]) * object_mat.inverted_affine();
  mul_m4_m4m4(persmat, winmat.values, viewmat.values);
}

void ShadowTileMap::debug_draw(void) const
{
  /** Used for debug drawing. */
  float4 debug_color[6] = {
      {1, .1, .1, 1}, {.1, 1, .1, 1}, {0, .2, 1, 1}, {1, 1, .3, 1}, {.1, .1, .1, 1}, {1, 1, 1, 1}};
  float4 color = debug_color[((is_cubeface ? cubeface : level) + 9999) % 6];

  float4x4 persinv = float4x4(persmat).inverted();
  DRW_debug_m4_as_bbox(persinv.values, color, false);

  int64_t div = ShadowTileAllocator::maps_per_row;
  std::stringstream ss;
  ss << "[" << index % div << ":" << index / div << "]";
  std::string text = ss.str();

  float3 pos = float3(0.0f, 0.0f, (is_cubeface) ? 1.0f : 0.0f);
  mul_project_m4_v3(persinv.values, pos);

  uchar ucolor[4] = {
      uchar(255 * color.x), uchar(255 * color.y), uchar(255 * color.z), uchar(255 * color.w)};
  struct DRWTextStore *dt = DRW_text_cache_ensure();
  DRW_text_cache_add(dt, pos, text.c_str(), text.size(), 0, 0, DRW_TEXT_CACHE_GLOBALSPACE, ucolor);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Tile map allocator
 *
 * \{ */

ShadowTileAllocator::ShadowTileAllocator()
{
  for (auto &bit : usage_bitmap_) {
    bit = false;
  }

  /* Try to not have a very long texture since it is
   * potentially wasteful on most GPU using tiled memory. */
  int tilemap_res = ShadowTileMap::tile_map_resolution;
  int2 extent;
  extent.x = min_ii(size, maps_per_row) * tilemap_res;
  extent.y = (size / maps_per_row) * tilemap_res;
  int mips = log2_ceil_u(tilemap_res);
  tilemap_tx.ensure(UNPACK2(extent), mips, GPU_R32UI);

  /* Allocate one pixel for 8 tilemap. */
  extent.x = size / 8;
  tilemap_update_result_tx.ensure(UNPACK2(extent), 1, GPU_R8UI);
}

ShadowTileAllocator::~ShadowTileAllocator()
{
  for (ShadowTileMap *map : maps) {
    delete map;
  }
}

/** Returns empty span on failure. */
Span<ShadowTileMap *> ShadowTileAllocator::alloc(int64_t count)
{
  int64_t candidate = -1;
  for (int64_t j = 0; j < size; j++) {
    int64_t i = (next_index + j) % size;
    if (usage_bitmap_[i] == false) {
      if (candidate == -1) {
        candidate = i;
      }
      if (i - candidate + 1 == count) {
        int64_t start = maps.size();
        for (auto j : IndexRange(candidate, count)) {
          usage_bitmap_[j] = true;
          maps.append(new ShadowTileMap(j));
        }
        next_index = candidate + count;
        return maps.as_span().slice(IndexRange(start, count));
      }
    }
    else {
      candidate = -1;
    }
  }
  return Span<ShadowTileMap *>();
}

void ShadowTileAllocator::free(Vector<ShadowTileMap *> &free_list)
{
  for (ShadowTileMap *map : free_list) {
    maps.remove_first_occurrence_and_reorder(map);
    usage_bitmap_[map->index] = false;
    maps_deleted.append(map);
    /* Actual deletion happens in end_sync(). */
  }
  free_list.clear();
}

void ShadowTileAllocator::end_sync()
{
  updated_maps_count = 0;
  for (ShadowTileMap *map : maps) {
    tilemaps_data[updated_maps_count++] = *map;
  }

  deleted_maps_count = 0;
  for (ShadowTileMap *map : maps_deleted) {
    /* Push to the ShadowTileMapsDataBuf in order to release the tiles.
     * Only do that if the slot was not reused for another map. */
    if (usage_bitmap_[map->index] == false) {
      /* This will effectively release all pages since they will be marked to update but not
       * marked as visible. */
      map->set_dirty();
      tilemaps_data[deleted_maps_count++] = *map;
    }
    delete map;
  }
  maps_deleted.clear();

  tilemaps_data.push_update();

  for (ShadowTileMap *map : maps) {
    map->set_updated();
    map->debug_draw();
  }
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Shadow Common
 *
 * \{ */

void ShadowCommon::free_resources()
{
  shadows_->tilemap_allocator.free(tilemaps);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Shadow Punctual
 *
 * \{ */

void ShadowPunctual::sync(eLightType light_type,
                          const mat4 &object_mat,
                          float cone_aperture,
                          float near_clip,
                          float far_clip,
                          float bias)
{
  bool is_wide_cone = cone_aperture > DEG2RAD(90);
  bool is_omni = cone_aperture > DEG2RAD(180);

  cone_aperture_ = cone_aperture;
  far_ = max_ff(far_clip, 3e-4f);
  near_ = min_ff(near_clip, far_clip - 1e-4f);
  bias_ = bias;
  light_type_ = light_type;

  /* Keep custom data. */
  size_x_ = _area_size_x;
  size_y_ = _area_size_y;

  position_ = float3(object_mat[3]);
  random_offset_ = float3(0.0f);

  int face_needed = is_omni ? 6 : (is_wide_cone ? 5 : 1);
  if (tilemaps.size() != face_needed) {
    shadows_->tilemap_allocator.free(tilemaps);
    tilemaps = shadows_->tilemap_allocator.alloc(face_needed);
  }

  float4x4 winmat;
  cubeface_winmat_get(winmat.values, near_, far_);

  tilemaps[Z_NEG]->sync_cubeface(object_mat, winmat, near_, far_, Z_NEG);
  if (is_wide_cone) {
    tilemaps[X_POS]->sync_cubeface(object_mat, winmat, near_, far_, X_POS);
    tilemaps[X_NEG]->sync_cubeface(object_mat, winmat, near_, far_, X_NEG);
    tilemaps[Y_POS]->sync_cubeface(object_mat, winmat, near_, far_, Y_POS);
    tilemaps[Y_NEG]->sync_cubeface(object_mat, winmat, near_, far_, Y_NEG);
  }
  if (is_omni) {
    tilemaps[Z_POS]->sync_cubeface(object_mat, winmat, near_, far_, Z_POS);
  }
}

ShadowPunctual::operator ShadowData()
{
  ShadowData data;
  cubeface_winmat_get(data.mat, near_, far_);
  invert_m4(data.mat);
  data.offset = random_offset_;
  data.bias = bias_;
  data.clip_near = near_;
  data.clip_far = far_;
  data.tilemap_index = tilemaps.first()->index;
  data.tilemap_last = data.tilemap_index + tilemaps.size() - 1;
  return data;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Directional Shadow Maps
 *
 * \{ */

void ShadowDirectional::sync(const mat4 &object_mat, float bias, float min_resolution)
{
  object_mat_ = float4x4(object_mat);
  min_resolution_ = min_resolution;
  bias_ = bias;
}

void ShadowDirectional::end_sync(const Camera &camera, const AABB &casters_bounds)
{
  int user_min_level = floorf(log2(min_resolution_));

  /* FIXME(fclem): We center the clipmap around the camera position which is arbitrary and
   * can affect lightprobe shadowing quality. To fix, just change camera position during bake and
   * profit!!! */

  /* Transpose is inverse if using only the 3x3 portion and the basis is orthogonal. */
  float4x4 obinv = object_mat_.transposed();

  AABB local_bounds;
  local_bounds.init_min_max();
  BoundBox bbox = casters_bounds;
  for (auto i : IndexRange(8)) {
    float3 local_vec = obinv.ref_3x3() * float3(bbox.vec[i]);
    local_bounds.merge(local_vec);
  }

  /* FIXME(fclem): This does not work in orthographic view. */
  int max_level = ceil(log2(fabsf(camera.data_get().clip_far)));
  int min_level = floor(log2(fabsf(camera.data_get().clip_near)));
  min_level = clamp_i(user_min_level, min_level, max_level);
  int level_count = max_level - min_level + 1;

  if (tilemaps.size() != level_count) {
    shadows_->tilemap_allocator.free(tilemaps);
    tilemaps = shadows_->tilemap_allocator.alloc(level_count);
  }

  /* Choose clipmap configuration. */
  /* TODO(fclem): We might want to improve / simplify the orthographic projection case since
   * all tiles would need the same resolution. The current clipmap distribution makes farthest
   * tiles less granular and potentially wasteful. */

  int level = min_level;
  for (ShadowTileMap *tilemap : tilemaps) {
    tilemap->sync_clipmap(camera, object_mat_, local_bounds, level++);
  }
}

ShadowDirectional::operator ShadowData()
{
  ShadowData data;
  invert_m4_m4(data.mat, object_mat_.values);
  data.bias = bias_;
  data.clip_near = near_;
  data.clip_far = far_;
  data.tilemap_index = tilemaps.first()->index;
  data.tilemap_last = data.tilemap_index + tilemaps.size() - 1;
  data.clipmap_lod_min = min_resolution_;
  data.clipmap_lod_max = min_resolution_ + tilemaps.size() - 1;
  return data;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Shadow Module
 *
 * \{ */

void ShadowModule::init(void)
{
  /* TODO(fclem) New resolution parameter. */
  // if (cube_shadow_res_ != inst_.scene->eevee.shadow_cube_size) {
  //   inst_.sampling.reset();
  // }

  eGPUTextureFormat shadow_format = (inst_.scene->eevee.flag & SCE_EEVEE_SHADOW_HIGH_BITDEPTH) ?
                                        GPU_DEPTH_COMPONENT32F :
                                        GPU_DEPTH_COMPONENT16;
  if (shadow_format_ != shadow_format) {
    shadow_format_ = shadow_format;
    inst_.sampling.reset();

    int2 atlas_extent = int2(shadow_page_size_ * ShadowTileMap::tile_map_resolution);
    GPUTexture *tex = atlas_tx_;

    /* Global update. */
    if ((tex == nullptr) || GPU_texture_format(atlas_tx_) != shadow_format_ ||
        GPU_texture_width(atlas_tx_) != atlas_extent.x ||
        GPU_texture_height(atlas_tx_) != atlas_extent.y) {
      for (ShadowTileMap *tilemap : tilemap_allocator.maps) {
        tilemap->set_dirty();
      }
    }

    /* TODO(fclem) Make allocation safe. */
    atlas_tx_.ensure(UNPACK2(atlas_extent), 1, shadow_format_);
    atlas_tx_ptr_ = atlas_tx_;
    /* Temporary render buffer. */
    render_tx_.ensure(UNPACK2(atlas_extent), 1, shadow_format_);
    render_fb_.ensure(GPU_ATTACHMENT_TEXTURE(render_tx_));

    tagging_tx_.ensure(16, 16, 1, GPU_DEPTH_COMPONENT16);
    tagging_fb_.ensure(GPU_ATTACHMENT_TEXTURE(tagging_tx_));
  }

  const bool soft_shadow_enabled = (inst_.scene->eevee.flag & SCE_EEVEE_SHADOW_SOFT) != 0;
  if (soft_shadows_enabled_ != soft_shadow_enabled) {
    soft_shadows_enabled_ = soft_shadow_enabled;
    inst_.sampling.reset();
  }

  if (G.debug_value == 4) {
    debug_data_.type = SHADOW_DEBUG_TILEMAPS;
  }

  memset(views_, 0, sizeof(views_));
}

void ShadowModule::begin_sync(void)
{
  casters_bounds_.init_min_max();
  receivers_non_opaque_ = DRW_call_buffer_create(&aabb_format_);
  casters_updated_ = DRW_call_buffer_create(&aabb_format_);
}

void ShadowModule::sync_object(Object *ob,
                               const ObjectHandle &handle,
                               bool is_shadow_caster,
                               bool is_alpha_blend)
{
  if (!is_shadow_caster && !is_alpha_blend) {
    return;
  }

  ShadowObject &shadow_ob = objects_.lookup_or_add_default(handle.object_key);
  shadow_ob.used = true;
  if (handle.recalc != 0 || !shadow_ob.initialized) {
    if (is_shadow_caster && shadow_ob.initialized) {
      DRW_buffer_add_entry_struct(casters_updated_, &shadow_ob.aabb);
    }
    shadow_ob.sync(ob);
    if (is_shadow_caster) {
      DRW_buffer_add_entry_struct(casters_updated_, &shadow_ob.aabb);
    }
  }

  if (is_shadow_caster) {
    casters_bounds_.merge(shadow_ob.aabb);
  }

  if (is_alpha_blend) {
    DRW_buffer_add_entry_struct(receivers_non_opaque_, &shadow_ob.aabb);
  }
}

void ShadowModule::end_sync(void)
{
  /* Search for deleted or updated shadow casters */
  Vector<ObjectKey, 0> deleted_keys;
  for (auto item : objects_.items()) {
    ShadowObject &shadow_ob = item.value;
    if (!shadow_ob.used) {
      deleted_keys.append(item.key);
      /* May not be a caster, but it does not matter, be conservative. */
      DRW_buffer_add_entry_struct(casters_updated_, &shadow_ob.aabb);
    }
    else {
      /* Clear for next sync. */
      shadow_ob.used = false;
    }
  }
  for (auto key : deleted_keys) {
    objects_.remove(key);
  }
  if (deleted_keys.size() > 0) {
    inst_.sampling.reset();
  }

  /* WARNING: Fragile, use same value as INIT_MINMAX. */
  bool no_casters = (casters_bounds_.min.x == 1.0e-30f);
  if (no_casters) {
    /* Avoid problems down the road. */
    casters_bounds_ = AABB(1.0f);
  }

  /* Finish setting up the tilemaps. */
  punctuals.resize();
  directionals.resize();

  for (ShadowDirectional &directional : directionals) {
    directional.end_sync(inst_.camera, casters_bounds_);
  }

  tilemap_allocator.end_sync();

  /**
   * Create update dispatches.
   */
  /* This is a workaround for DRW_STATE_RASTERIZER_ENABLED being a thing.
   * Can be removed when it will be gone. The framebuffer has no color attachments. */
  DRWState no_raster_discard = DRW_STATE_WRITE_COLOR;

  {
    /* Reset usage bit and do shifting if any. */
    tilemap_setup_ps_ = DRW_pass_create("ShadowTilemapSetup", (DRWState)0);

    GPUShader *sh = inst_.shaders.static_shader_get(SHADOW_TILE_SETUP);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, tilemap_setup_ps_);
    DRW_shgroup_vertex_buffer(grp, "tilemaps_block", tilemap_allocator.tilemaps_data);
    DRW_shgroup_uniform_image(grp, "tilemaps_img", tilemap_allocator.tilemap_tx);
    int64_t tilemaps_data_len = tilemap_allocator.updated_maps_count +
                                tilemap_allocator.deleted_maps_count;
    if (tilemaps_data_len > 0) {
      DRW_shgroup_call_compute(grp, 1, 1, tilemaps_data_len);
      DRW_shgroup_barrier(grp, GPU_BARRIER_SHADER_IMAGE_ACCESS);
    }
    tilemap_setup_has_run_ = false;
  }
  {
    tilemap_visibility_ps_ = DRW_pass_create("ShadowVisibilityTag", (DRWState)0);

    GPUShader *sh = inst_.shaders.static_shader_get(SHADOW_TILE_TAG_VISIBILITY);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, tilemap_visibility_ps_);
    DRW_shgroup_vertex_buffer(grp, "tilemaps_block", tilemap_allocator.tilemaps_data);
    DRW_shgroup_uniform_image(grp, "tilemaps_img", tilemap_allocator.tilemap_tx);
    int64_t tilemaps_data_len = tilemap_allocator.updated_maps_count;
    if (tilemaps_data_len > 0) {
      DRW_shgroup_call_compute(grp, 1, 1, tilemaps_data_len);
      DRW_shgroup_barrier(grp, GPU_BARRIER_SHADER_IMAGE_ACCESS);
    }
  }
  {
    /* Set usage bit using load/store. */
    tilemap_usage_tag_ps_ = DRW_pass_create("ShadowUsageTag", no_raster_discard);

    GPUShader *sh = inst_.shaders.static_shader_get(SHADOW_TILE_TAG_USAGE);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, tilemap_usage_tag_ps_);
    DRW_shgroup_vertex_buffer(grp, "tilemaps_block", tilemap_allocator.tilemaps_data);
    DRW_shgroup_uniform_image(grp, "tilemaps_img", tilemap_allocator.tilemap_tx);
    int64_t tilemaps_data_len = tilemap_allocator.updated_maps_count;
    if (tilemaps_data_len > 0) {
      DRW_shgroup_call_buffer_ex(grp, GPU_PRIM_POINTS, casters_updated_, tilemaps_data_len);
      DRW_shgroup_barrier(grp, GPU_BARRIER_SHADER_IMAGE_ACCESS);
    }

    /* TODO(fclem): Add debug buffer scanning here. */
  }
  {
    /* Set update bit using load/store. */
    tilemap_update_tag_ps_ = DRW_pass_create("ShadowUpdateTag", no_raster_discard);

    GPUShader *sh = inst_.shaders.static_shader_get(SHADOW_TILE_TAG_UPDATE);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, tilemap_update_tag_ps_);
    DRW_shgroup_vertex_buffer(grp, "tilemaps_block", tilemap_allocator.tilemaps_data);
    DRW_shgroup_uniform_image(grp, "tilemaps_img", tilemap_allocator.tilemap_tx);
    int64_t tilemaps_data_len = tilemap_allocator.updated_maps_count;
    if (tilemaps_data_len > 0) {
      DRW_shgroup_call_buffer_ex(grp, GPU_PRIM_POINTS, casters_updated_, tilemaps_data_len);
      DRW_shgroup_barrier(grp, GPU_BARRIER_SHADER_IMAGE_ACCESS);
    }
  }

  debug_end_sync();
}

void ShadowModule::debug_end_sync(void)
{
  debug_draw_ps_ = nullptr;

  if (debug_data_.type == SHADOW_DEBUG_NONE) {
    return;
  }

  Object *obact = DRW_context_state_get()->obact;
  if (obact && (obact->type == OB_LAMP)) {
    /* Dangerous. But only used for debug. */
    debug_light_key = inst_.sync.sync_object(obact).object_key;
  }

  if (debug_light_key.ob == nullptr) {
    return;
  }

  LightModule &light_module = inst_.lights;
  if (light_module.lights_.contains(debug_light_key) == false) {
    return;
  }
  Light &light = light_module.lights_.lookup(debug_light_key);
  if (light.shadow_id == LIGHT_NO_SHADOW) {
    return;
  }

  debug_data_.light = light;
  if (light.type == LIGHT_SUN) {
    debug_data_.shadow = directionals[light.shadow_id];
  }
  else {
    debug_data_.shadow = punctuals[light.shadow_id];
  }
  debug_data_.push_update();

  {
    DRWState state = DRW_STATE_WRITE_COLOR | DRW_STATE_WRITE_DEPTH | DRW_STATE_DEPTH_LESS |
                     DRW_STATE_BLEND_CUSTOM;
    debug_draw_ps_ = DRW_pass_create("ShadowDebugDraw", state);

    GPUShader *sh = inst_.shaders.static_shader_get(SHADOW_DEBUG);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, debug_draw_ps_);
    DRW_shgroup_uniform_texture(grp, "tilemaps_tx", tilemap_allocator.tilemap_tx);
    DRW_shgroup_uniform_texture_ref(grp, "depth_tx", &input_depth_tx_);
    DRW_shgroup_uniform_block(grp, "debug_block", debug_data_.ubo_get());
    DRW_shgroup_call_procedural_triangles(grp, nullptr, 1);
  }
}

/* Update all shadow regions visible inside the view. */
void ShadowModule::update_visible(const DRWView *view)
{
#if 0 /* TODO */
  bool force_update = false;
  if (soft_shadows_enabled_ && (inst_.sampling.sample_get() != last_sample_)) {
    force_update = true;
    last_sample_ = inst_.sampling.sample_get();
  }
  else {
    last_sample_ = 0;
  }
#endif

  DRW_view_set_active(view);

  DRW_stats_group_start("ShadowUpdate");
  {

    /* Setup only once. */
    if (tilemap_setup_has_run_ == false) {
      tilemap_setup_has_run_ = true;
      DRW_draw_pass(tilemap_setup_ps_);
    }
    DRW_draw_pass(tilemap_visibility_ps_);

    GPU_framebuffer_bind(tagging_fb_);
    DRW_draw_pass(tilemap_usage_tag_ps_);
    DRW_draw_pass(tilemap_update_tag_ps_);

    /**
     * This is the most complex part in the entire shadow pipeline.
     * This step will read each updated tilemap to see if any tile is both visible and to be
     * updated. If that is the case, it computes the bounds of the tiles to update and write it
     * in a texture to be read back by the CPU. This is a sync step that is the main performance
     * bottleneck of the pipeline.
     *
     * Unused tile might be reallocated at this stage.
     *
     * For each unallocated tile it will reserve a new page in the atlas. If the tile is to be
     * rendered, it will also write the tile copy coordinates required in another buffer.
     * This is also a slow part and should be improved in the future by moving the least amount of
     * tiles.
     */
    // DRW_draw_pass(tilemap_scheduling_ps_);

    /* Readback update list. Ugly sync point. */
  }
  DRW_stats_group_end();

  DRW_stats_group_start("ShadowRender");
  {
#if 0
    for (tilemap : tilemap_to_render) {
      /* TODO(fclem): Setup DRWView to fit min AABB. */
      DRWView *shadow_view = tilemap.draw_view_get();
      DRW_view_set_active(shadow_view);

      GPU_framebuffer_bind(render_fb_);

      /* TODO(fclem): When not using rendering to another texture, tag tiles to not touch.
       * Can also reduce complex pixel shader workload. */
      // DRW_draw_pass(tile_tag_stencil_ps_);

      inst_.shading_passes.shadow.render();

      /* Copy result to the shadow atlas. */
      /* TODO(fclem) avoid this pass as much as we can. */
      tile_copy_start_ = tilemap_update.tile_copy_start;
      tile_copy_end_ = tilemap_update.tile_copy_end;
      DRW_draw_pass(tiles_copy_ps_);
    }
#endif
  }
  DRW_stats_group_end();

  DRW_view_set_active(view);
}

void ShadowModule::debug_draw(GPUFrameBuffer *view_fb, GPUTexture *depth_tx)
{
  if (debug_draw_ps_ == nullptr) {
    return;
  }
  input_depth_tx_ = depth_tx;

  GPU_framebuffer_bind(view_fb);
  DRW_draw_pass(debug_draw_ps_);
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
     * - Dedicated Image View & framebuffer for each AtlasRegion.
     */
    DRWState state = DRW_STATE_WRITE_DEPTH | DRW_STATE_DEPTH_ALWAYS;
    clear_ps_ = DRW_pass_create("ShadowClear", state);

    GPUShader *sh = inst_.shaders.static_shader_get(SHADOW_CLEAR);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, clear_ps_);
    DRW_shgroup_call_procedural_triangles(grp, nullptr, 1);
  }
  {
    DRWState state = DRW_STATE_WRITE_DEPTH | DRW_STATE_DEPTH_LESS | DRW_STATE_SHADOW_OFFSET;
    surface_ps_ = DRW_pass_create("ShadowSurface", state);
  }
}

DRWShadingGroup *ShadowPass::material_add(::Material *UNUSED(material), GPUMaterial *gpumat)
{
  DRWShadingGroup *grp = DRW_shgroup_material_create(gpumat, surface_ps_);
  DRW_shgroup_uniform_block(grp, "sampling_block", inst_.sampling.ubo_get());
  return grp;
}

void ShadowPass::render(void)
{
  DRW_draw_pass(clear_ps_);
  DRW_draw_pass(surface_ps_);
}

/** \} */

}  // namespace blender::eevee
