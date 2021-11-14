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
#include "BLI_rect.h"

#include "eevee_instance.hh"

#include "draw_manager_text.h"

#include <iostream>

namespace blender::eevee {

/* -------------------------------------------------------------------- */
/** \name Tile map
 *
 * \{ */

void ShadowTileMap::sync_clipmap(const float3 &camera_position,
                                 const float4x4 &object_mat_,
                                 float near_,
                                 float far_,
                                 int clipmap_level)
{
#ifdef SHADOW_NO_CACHING
  set_dirty();
#endif
  if (is_cubeface || (level != clipmap_level) || (near != near_) || (far != far_)) {
    set_dirty();
  }
  is_cubeface = false;
  level = clipmap_level;
  near = near_;
  far = far_;
  cone_direction = float3(1.0f);
  cone_angle_cos = -2.0f;

  float4x4 object_mat = object_mat_;
  /* Clear embedded custom data. */
  object_mat.values[0][3] = object_mat.values[1][3] = object_mat.values[2][3] = 0.0f;
  object_mat.values[3][3] = 1.0f;
  /* Remove translation. */
  zero_v3(object_mat.values[3]);

  if (!equals_m4m4(obmat.values, object_mat.values)) {
    obmat = object_mat;
    set_dirty();
  }

  /* Same as object_mat.inverted() because object_mat it is orthogonal. */
  float3 tilemap_center = object_mat.transposed() * camera_position;

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
  if (grid_shift == int2(0)) {
    /* Only replace shift if it is not already dirty. */
    grid_shift = new_offset - grid_offset;
  }
  grid_offset = new_offset;

  tilemap_center = object_mat * float3(grid_offset.x, grid_offset.y, 0.0f);

  copy_v3_v3(object_mat.values[3], tilemap_center);

  float half_size = tilemap_coverage_get() / 2.0f;
  orthographic_m4(winmat.values, -half_size, half_size, -half_size, half_size, near, far);
  winmat = winmat_get(nullptr);
  viewmat = object_mat.inverted_affine();

  mul_m4_m4m4(persmat, winmat.values, viewmat.values);

  /* Update corners. */
  float4x4 viewinv = viewmat.inverted();
  *reinterpret_cast<float3 *>(&corners[0]) = viewinv * float3(-half_size, -half_size, near);
  *reinterpret_cast<float3 *>(&corners[1]) = viewinv * float3(half_size, -half_size, near);
  *reinterpret_cast<float3 *>(&corners[2]) = viewinv * float3(-half_size, half_size, near);
  *reinterpret_cast<float3 *>(&corners[3]) = viewinv * float3(-half_size, -half_size, far);
  /* Store deltas. */
  corners[1] = (corners[1] - corners[0]) / float(SHADOW_TILEMAP_RES);
  corners[2] = (corners[2] - corners[0]) / float(SHADOW_TILEMAP_RES);
  corners[3] -= corners[0];
}

void ShadowTileMap::sync_cubeface(
    const float4x4 &object_mat_, float near_, float far_, float cone_aperture, eCubeFace face)
{
#ifdef SHADOW_NO_CACHING
  set_dirty();
#endif
  if (!is_cubeface || (cubeface != face) || (near != near_) || (far != far_)) {
    set_dirty();
  }
  is_cubeface = true;
  cubeface = face;
  near = near_;
  far = far_;

  if (cone_aperture > DEG2RADF(180.0f)) {
    cone_angle_cos = -2.0;
  }
  else {
    cone_angle_cos = cosf(min_ff((cone_aperture * 0.5f) + tile_cone_half_angle, M_PI_2));
  }
  cone_direction = -float3(object_mat_.values[2]);

  float4x4 object_mat = object_mat_;
  /* Clear embedded custom data. */
  object_mat.values[0][3] = object_mat.values[1][3] = object_mat.values[2][3] = 0.0f;
  object_mat.values[3][3] = 1.0f;

  if (obmat != object_mat) {
    obmat = object_mat;
    set_dirty();
  }

  viewmat = float4x4(shadow_face_mat[cubeface]) * object_mat.inverted_affine();
  winmat = winmat_get(nullptr);

  mul_m4_m4m4(persmat, winmat.values, viewmat.values);

  /* Update corners. */
  float4x4 viewinv = viewmat.inverted();
  *reinterpret_cast<float3 *>(&corners[0]) = viewinv.translation();
  *reinterpret_cast<float3 *>(&corners[1]) = viewinv * float3(-far, -far, -far);
  *reinterpret_cast<float3 *>(&corners[2]) = viewinv * float3(far, -far, -far);
  *reinterpret_cast<float3 *>(&corners[3]) = viewinv * float3(-far, far, -far);
  /* Store deltas. */
  corners[2] = (corners[2] - corners[1]) / float(SHADOW_TILEMAP_RES);
  corners[3] = (corners[3] - corners[1]) / float(SHADOW_TILEMAP_RES);
}

float4x4 ShadowTileMap::winmat_get(const rcti *tile_minmax) const
{
  float2 min = float2(-1.0f);
  float2 max = float2(1.0f);

  if (tile_minmax != nullptr) {
    min = shadow_tile_coord_to_ndc(int2(tile_minmax->xmin + 0, tile_minmax->ymin + 0));
    max = shadow_tile_coord_to_ndc(int2(tile_minmax->xmax + 1, tile_minmax->ymax + 1));
  }

  float4x4 winmat;
  if (is_cubeface) {
    perspective_m4(
        winmat.values, near * min.x, near * max.x, near * min.y, near * max.y, near, far);
  }
  else {
    float half_size = tilemap_coverage_get() / 2.0f;
    orthographic_m4(winmat.values,
                    half_size * min.x,
                    half_size * max.x,
                    half_size * min.y,
                    half_size * max.y,
                    near,
                    far);
  }
  return winmat;
}

void ShadowTileMap::setup_view(const rcti &rect, DRWView *&view) const
{
  float4x4 culling_mat = winmat_get(&rect);

  if (view == nullptr) {
    view = DRW_view_create(viewmat.values, winmat.values, nullptr, culling_mat.values, nullptr);
  }
  else {
    DRW_view_update(view, viewmat.values, winmat.values, nullptr, culling_mat.values);
  }

#if 0 /* Debug. */
  float4 debug_color[6] = {
      {1, .1, .1, 1}, {.1, 1, .1, 1}, {0, .2, 1, 1}, {1, 1, .3, 1}, {.1, .1, .1, 1}, {1, 1, 1, 1}};
  float4 color = debug_color[((is_cubeface ? cubeface : level) + 9999) % 6];
  float4x4 persinv_culling = (culling_mat * viewmat).inverted();
  DRW_debug_m4_as_bbox(persinv_culling.values, color, false);
#endif
}

void ShadowTileMap::debug_draw(void) const
{
  /** Used for debug drawing. */
  float4 debug_color[6] = {
      {1, .1, .1, 1}, {.1, 1, .1, 1}, {0, .2, 1, 1}, {1, 1, .3, 1}, {.1, .1, .1, 1}, {1, 1, 1, 1}};
  float4 color = debug_color[((is_cubeface ? cubeface : level) + 9999) % 6];

  float persinv[4][4];
  invert_m4_m4(persinv, persmat);
  DRW_debug_m4_as_bbox(persinv, color, false);

  int64_t div = ShadowTileAllocator::maps_per_row;
  std::stringstream ss;
  ss << "[" << index % div << ":" << index / div << "]";
  std::string text = ss.str();

  float3 pos = float3(0.0f, 0.0f, (is_cubeface) ? 1.0f : 0.0f);
  mul_project_m4_v3(persinv, pos);

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
  tilemap_tx.clear((uint)0);

  /* Allocate one pixel for each tilemap. */
  tilemap_rects_tx.ensure(size, 1, 1, GPU_RGBA32I);
  tilemap_rects_tx.clear((int)0);
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
  /* Iterate through the whole buffer starting from the last known alloc position. */
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
  active_maps_len = 0;
  for (ShadowTileMap *map : maps) {
    tilemaps_data[active_maps_len++] = *map;
  }

  deleted_maps_len = 0;
  for (ShadowTileMap *map : maps_deleted) {
    /* Push to the ShadowTileMapsDataBuf in order to release the tiles.
     * Only do that if the slot was not reused for another map. */
    if (usage_bitmap_[map->index] == false) {
      /* This will effectively release all pages since they will be marked to update but not
       * marked as visible. */
      map->set_dirty();
      tilemaps_data[active_maps_len + deleted_maps_len++] = *map;
    }
    delete map;
  }
  maps_deleted.clear();

  tilemaps_data.push_update();

  for (ShadowTileMap *map : maps) {
    map->set_updated();
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
  bool is_wide_cone = cone_aperture > DEG2RADF(90.0f);
  bool is_omni = cone_aperture > DEG2RADF(180.0f);

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

  tilemaps[Z_NEG]->sync_cubeface(object_mat, near_, far_, cone_aperture, Z_NEG);
  if (is_wide_cone) {
    tilemaps[X_POS]->sync_cubeface(object_mat, near_, far_, cone_aperture, X_POS);
    tilemaps[X_NEG]->sync_cubeface(object_mat, near_, far_, cone_aperture, X_NEG);
    tilemaps[Y_POS]->sync_cubeface(object_mat, near_, far_, cone_aperture, Y_POS);
    tilemaps[Y_NEG]->sync_cubeface(object_mat, near_, far_, cone_aperture, Y_NEG);
  }
  if (is_omni) {
    tilemaps[Z_POS]->sync_cubeface(object_mat, near_, far_, cone_aperture, Z_POS);
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

void ShadowDirectional::end_sync(int min_level,
                                 int max_level,
                                 const float3 &camera_position,
                                 const AABB &casters_bounds)
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

  float near = -local_bounds.max.z + 1e-8f;
  float far = -local_bounds.min.z - 1e-8f;

  int level = min_level;
  for (ShadowTileMap *tilemap : tilemaps) {
    tilemap->sync_clipmap(camera_position, object_mat_, near, far, level++);
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

    int2 atlas_extent = int2(shadow_page_size_ * SHADOW_PAGE_PER_ROW);
    int2 render_extent = int2(shadow_page_size_ * SHADOW_TILEMAP_RES);
    GPUTexture *tex = atlas_tx_;

    /* Global update. */
    if ((tex == nullptr) || GPU_texture_format(atlas_tx_) != shadow_format_ ||
        GPU_texture_width(atlas_tx_) != atlas_extent.x ||
        GPU_texture_height(atlas_tx_) != atlas_extent.y) {
      for (ShadowTileMap *tilemap : tilemap_allocator.maps) {
        tilemap->set_dirty();
      }
    }

    /* TODO(fclem) GPU_DEPTH_COMPONENT16 support in copy shader? */
    /* TODO(fclem) Make allocation safe. */
    atlas_tx_.ensure(UNPACK2(atlas_extent), 1, GPU_R32F);
    atlas_tx_ptr_ = atlas_tx_;
#if DEBUG
    atlas_tx_.clear(0.0f);
#endif
    /* Temporary render buffer. */
    render_tx_.ensure(UNPACK2(render_extent), 1, GPU_DEPTH_COMPONENT32F);
    render_fb_.ensure(GPU_ATTACHMENT_TEXTURE(render_tx_));
  }

  const bool soft_shadow_enabled = (inst_.scene->eevee.flag & SCE_EEVEE_SHADOW_SOFT) != 0;
  if (soft_shadows_enabled_ != soft_shadow_enabled) {
    soft_shadows_enabled_ = soft_shadow_enabled;
    inst_.sampling.reset();
  }

  switch (G.debug_value) {
    case 4:
      debug_data_.type = SHADOW_DEBUG_TILEMAPS;
      break;
    case 5:
      debug_data_.type = SHADOW_DEBUG_LOD;
      break;
    case 6:
      debug_data_.type = SHADOW_DEBUG_PAGE_ALLOCATION;
#ifndef SHADOW_DEBUG_PAGE_ALLOCATION_ENABLED
      BLI_assert_msg(0,
                     "Error: EEVEE: SHADOW_DEBUG_PAGE_ALLOCATION used but "
                     "SHADOW_DEBUG_PAGE_ALLOCATION_ENABLED "
                     "is not defined");
#endif
      break;
    case 7:
      debug_data_.type = SHADOW_DEBUG_TILE_ALLOCATION;
      break;
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

  {
    /* Get the farthest point from camera to know what distance to cover. */
    float3 farthest_point = float3(1.0f, 1.0f, 1.0f);
    mul_project_m4_v3(inst_.camera.data_get().wininv, farthest_point);
    float far_dist = farthest_point.length();
    float near_dist = fabsf(inst_.camera.data_get().clip_near);
    float3 cam_position = inst_.camera.position();

#ifdef DEBUG
    static bool valid = false;
    static float far_dist_freezed;
    static float near_dist_freezed;
    static float3 cam_position_freezed;
    if (G.debug_value < 4 || !valid) {
      valid = true;
      far_dist_freezed = far_dist;
      near_dist_freezed = near_dist;
      cam_position_freezed = cam_position;
    }
    else {
      far_dist = far_dist_freezed;
      near_dist = near_dist_freezed;
      cam_position = cam_position_freezed;
    }
#endif
    /* FIXME(fclem): This does not work in orthographic view. */
    int max_level = ceil(log2(far_dist));
    int min_level = floor(log2(near_dist));

    for (ShadowDirectional &directional : directionals) {
      directional.end_sync(min_level, max_level, cam_position, casters_bounds_);
    }
  }

  tilemap_allocator.end_sync();

  /**
   * Tilemap Management
   */

  int64_t tilemaps_len = tilemap_allocator.active_maps_len;
  {
    tilemap_setup_ps_ = DRW_pass_create("ShadowTilemapSetup", (DRWState)0);

    GPUShader *sh = inst_.shaders.static_shader_get(SHADOW_TILE_SETUP);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, tilemap_setup_ps_);
    DRW_shgroup_vertex_buffer(grp, "tilemaps_buf", tilemap_allocator.tilemaps_data);
    DRW_shgroup_uniform_image(grp, "tilemaps_img", tilemap_allocator.tilemap_tx);
    int64_t tilemaps_updated_len = tilemaps_len + tilemap_allocator.deleted_maps_len;
    if (tilemaps_updated_len > 0) {
      DRW_shgroup_call_compute(grp, 1, 1, tilemaps_updated_len);
      DRW_shgroup_barrier(grp, GPU_BARRIER_SHADER_IMAGE_ACCESS);
    }
    do_tilemap_setup_ = true;
  }
  {
    tilemap_visibility_ps_ = DRW_pass_create("ShadowVisibilityTag", (DRWState)0);

    GPUShader *sh = inst_.shaders.static_shader_get(SHADOW_TILE_TAG_VISIBILITY);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, tilemap_visibility_ps_);
    DRW_shgroup_vertex_buffer(grp, "tilemaps_buf", tilemap_allocator.tilemaps_data);
    DRW_shgroup_uniform_image(grp, "tilemaps_img", tilemap_allocator.tilemap_tx);
    if (tilemaps_len > 0) {
      DRW_shgroup_call_compute(grp, 1, 1, tilemaps_len);
      DRW_shgroup_barrier(grp, GPU_BARRIER_SHADER_IMAGE_ACCESS);
    }
  }
  {
    tilemap_usage_tag_ps_ = DRW_pass_create("ShadowUsageTag", (DRWState)0);

    GPUVertBuf *receivers_aabbs = DRW_call_buffer_as_vertbuf(receivers_non_opaque_);
    uint aabb_len = GPU_vertbuf_get_vertex_len(receivers_aabbs);

    GPUShader *sh = inst_.shaders.static_shader_get(SHADOW_TILE_TAG_USAGE);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, tilemap_usage_tag_ps_);
    DRW_shgroup_vertex_buffer(grp, "aabb_buf", receivers_aabbs);
    DRW_shgroup_vertex_buffer(grp, "tilemaps_buf", tilemap_allocator.tilemaps_data);
    DRW_shgroup_uniform_image(grp, "tilemaps_img", tilemap_allocator.tilemap_tx);
    DRW_shgroup_uniform_int_copy(grp, "aabb_len", aabb_len);
    if (tilemaps_len > 0 && aabb_len > 0) {
      uint group_len = divide_ceil_u(aabb_len, SHADOW_AABB_TAG_GROUP_SIZE);
      DRW_shgroup_call_compute(grp, group_len, 1, tilemaps_len);
      DRW_shgroup_barrier(grp, GPU_BARRIER_SHADER_IMAGE_ACCESS);
    }

    /* TODO(fclem): Add depth buffer scanning here. */
  }
  {
    tilemap_update_tag_ps_ = DRW_pass_create("ShadowUpdateTag", (DRWState)0);

    GPUVertBuf *casters_aabbs = DRW_call_buffer_as_vertbuf(casters_updated_);
    uint aabb_len = GPU_vertbuf_get_vertex_len(casters_aabbs);

    GPUShader *sh = inst_.shaders.static_shader_get(SHADOW_TILE_TAG_UPDATE);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, tilemap_update_tag_ps_);
    DRW_shgroup_vertex_buffer(grp, "aabb_buf", casters_aabbs);
    DRW_shgroup_vertex_buffer(grp, "tilemaps_buf", tilemap_allocator.tilemaps_data);
    DRW_shgroup_uniform_image(grp, "tilemaps_img", tilemap_allocator.tilemap_tx);
    DRW_shgroup_uniform_int_copy(grp, "aabb_len", GPU_vertbuf_get_vertex_len(casters_aabbs));
    if (tilemaps_len > 0 && aabb_len > 0) {
      uint group_len = divide_ceil_u(aabb_len, SHADOW_AABB_TAG_GROUP_SIZE);
      DRW_shgroup_call_compute(grp, group_len, 1, tilemaps_len);
      DRW_shgroup_barrier(grp, GPU_BARRIER_SHADER_IMAGE_ACCESS);
    }
  }
  {
    tilemap_propagate_lod_ps_ = DRW_pass_create("ShadowLodPropagate", (DRWState)0);
    /* TODO */
  }

  /**
   * Page Management
   */

  {
    page_init_ps_ = DRW_pass_create("ShadowPageInit", (DRWState)0);

    GPUShader *sh = inst_.shaders.static_shader_get(SHADOW_PAGE_INIT);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, page_init_ps_);
    DRW_shgroup_vertex_buffer(grp, "pages_free_buf", pages_free_data_);
    DRW_shgroup_vertex_buffer(grp, "pages_buf", pages_data_);
    DRW_shgroup_call_compute(grp, SHADOW_MAX_PAGE / SHADOW_PAGE_PER_ROW, 1, 1);
    DRW_shgroup_barrier(grp, GPU_BARRIER_SHADER_STORAGE);
  }
  {
    page_free_ps_ = DRW_pass_create("ShadowPageFree", (DRWState)0);

    GPUShader *sh = inst_.shaders.static_shader_get(SHADOW_PAGE_FREE);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, page_free_ps_);
    DRW_shgroup_vertex_buffer(grp, "pages_free_buf", pages_free_data_);
    DRW_shgroup_vertex_buffer(grp, "pages_buf", pages_data_);
    DRW_shgroup_vertex_buffer(grp, "tilemaps_buf", tilemap_allocator.tilemaps_data);
    DRW_shgroup_uniform_image(grp, "tilemaps_img", tilemap_allocator.tilemap_tx);
    int64_t tilemaps_updated_len = tilemaps_len + tilemap_allocator.deleted_maps_len;
    if (tilemaps_updated_len > 0) {
      DRW_shgroup_call_compute(grp, 1, 1, tilemaps_updated_len);
      DRW_shgroup_barrier(grp, GPU_BARRIER_SHADER_IMAGE_ACCESS);
      DRW_shgroup_barrier(grp, GPU_BARRIER_SHADER_STORAGE);
    }
  }
  {
    page_alloc_ps_ = DRW_pass_create("ShadowPageAllocate", (DRWState)0);

    GPUShader *sh = inst_.shaders.static_shader_get(SHADOW_PAGE_ALLOC);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, page_alloc_ps_);
    DRW_shgroup_vertex_buffer(grp, "pages_free_buf", pages_free_data_);
    DRW_shgroup_vertex_buffer(grp, "pages_buf", pages_data_);
    DRW_shgroup_vertex_buffer(grp, "tilemaps_buf", tilemap_allocator.tilemaps_data);
    DRW_shgroup_uniform_image(grp, "tilemaps_img", tilemap_allocator.tilemap_tx);
    DRW_shgroup_uniform_image(grp, "tilemap_rects_img", tilemap_allocator.tilemap_rects_tx);
    if (tilemaps_len > 0) {
      DRW_shgroup_call_compute(grp, 1, 1, tilemaps_len);
      DRW_shgroup_barrier(grp, GPU_BARRIER_SHADER_IMAGE_ACCESS);
      DRW_shgroup_barrier(grp, GPU_BARRIER_SHADER_STORAGE);
    }
  }
  {
    DRWState state = DRW_STATE_WRITE_DEPTH | DRW_STATE_DEPTH_ALWAYS;
    page_mark_ps_ = DRW_pass_create("ShadowPageMark", state);

    GPUShader *sh = inst_.shaders.static_shader_get(SHADOW_PAGE_MARK);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, page_mark_ps_);
    DRW_shgroup_uniform_texture(grp, "tilemaps_tx", tilemap_allocator.tilemap_tx);
    DRW_shgroup_uniform_int(grp, "tilemap_index", &rendering_tilemap_, 1);
    DRW_shgroup_clear_framebuffer(grp, GPU_DEPTH_BIT, 0, 0, 0, 0, 0.0f, 0x0);
    DRW_shgroup_call_procedural_triangles(grp, nullptr, square_i(SHADOW_TILEMAP_RES) * 2);
  }
  {
    page_copy_ps_ = DRW_pass_create("ShadowPageCopy", (DRWState)0);

    GPUShader *sh = inst_.shaders.static_shader_get(SHADOW_PAGE_COPY);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, page_copy_ps_);
    DRW_shgroup_uniform_texture(grp, "tilemaps_tx", tilemap_allocator.tilemap_tx);
    DRW_shgroup_uniform_texture(grp, "render_tx", render_tx_);
    DRW_shgroup_uniform_image(grp, "out_atlas_img", atlas_tx_);
    int dispatch_size = shadow_page_size_ / SHADOW_PAGE_COPY_GROUP_SIZE;
    DRW_shgroup_call_compute(grp, dispatch_size, dispatch_size, 1);
    DRW_shgroup_barrier(grp, GPU_BARRIER_SHADER_IMAGE_ACCESS);
  }

  debug_end_sync();
}

void ShadowModule::debug_end_sync(void)
{
  debug_draw_ps_ = nullptr;

  if (debug_data_.type == SHADOW_DEBUG_NONE) {
    return;
  }

  const bool need_active_light_data = !ELEM(
      debug_data_.type, SHADOW_DEBUG_TILE_ALLOCATION, SHADOW_DEBUG_PAGE_ALLOCATION);

  if (need_active_light_data) {
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
    /* Find index of tilemap data. */
    for (auto index : IndexRange(tilemap_allocator.size)) {
      if (debug_data_.shadow.tilemap_index == tilemap_allocator.tilemaps_data[index].index) {
        debug_data_.tilemap_data_index = index;
        break;
      }
    }
  }

  debug_data_.push_update();

  {
    DRWState state = DRW_STATE_WRITE_COLOR | DRW_STATE_WRITE_DEPTH | DRW_STATE_DEPTH_LESS_EQUAL |
                     DRW_STATE_BLEND_CUSTOM;
    debug_draw_ps_ = DRW_pass_create("ShadowDebugDraw", state);

    GPUShader *sh = inst_.shaders.static_shader_get(SHADOW_DEBUG);
    DRWShadingGroup *grp = DRW_shgroup_create(sh, debug_draw_ps_);
    DRW_shgroup_vertex_buffer(grp, "tilemaps_buf", tilemap_allocator.tilemaps_data);
    DRW_shgroup_vertex_buffer(grp, "pages_buf", pages_data_);
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

#ifdef DEBUG
  static bool valid = false;
  static float4x4 viewmat_freezed;
  if (G.debug_value < 4 || !valid) {
    valid = true;
    DRW_view_viewmat_get(view, viewmat_freezed.values, false);
  }
  else {
    float4x4 winmat;
    DRW_view_winmat_get(view, winmat.values, false);
    DRWView *debug_view = DRW_view_create(
        viewmat_freezed.values, winmat.values, nullptr, nullptr, nullptr);
    DRW_view_set_active(debug_view);

    float4 color(1.0f);
    float4x4 persinv;
    DRW_view_persmat_get(debug_view, persinv.values, true);
    DRW_debug_m4_as_bbox(persinv.values, color, false);
  }
#endif

  DRW_stats_group_start("ShadowUpdate");
  {
    if (do_tilemap_setup_) {
      do_tilemap_setup_ = false;
      DRW_draw_pass(tilemap_setup_ps_);
    }
    DRW_draw_pass(tilemap_visibility_ps_);
    DRW_draw_pass(tilemap_usage_tag_ps_);
    DRW_draw_pass(tilemap_update_tag_ps_);

    if (do_page_init_) {
#ifndef SHADOW_NO_CACHING
      do_page_init_ = false;
#endif
      DRW_draw_pass(page_init_ps_);
    }
    DRW_draw_pass(page_free_ps_);
    DRW_draw_pass(page_alloc_ps_);

    DRW_draw_pass(tilemap_propagate_lod_ps_);
  }
  DRW_stats_group_end();

  DRW_stats_group_start("ShadowRender");
  {
    /* Readback update list. Ugly sync point. */
    rcti *rect = tilemap_allocator.tilemap_rects_tx.read<rcti>(GPU_DATA_INT);

    Span<rcti> regions(rect, tilemap_allocator.maps.size());
    Span<ShadowTileMap *> tilemaps = tilemap_allocator.maps.as_span();

    while (!regions.is_empty()) {
      int region_len = 0;
      /* Group updates by pack of 6. This is to workaround the current DRWView limitation.
       * Future goal is to have GPU culling and create the views on GPU. */
      while (!regions.is_empty() && region_len < 6) {
        if (!BLI_rcti_is_empty(&regions.first())) {
          tilemaps.first()->setup_view(regions.first(), views_[region_len++]);
        }
        regions = regions.drop_front(1);
        tilemaps = tilemaps.drop_front(1);
      }

      for (auto i : IndexRange(region_len)) {
        DRW_view_set_active(views_[i]);
        GPU_framebuffer_bind(render_fb_);
        DRW_draw_pass(page_mark_ps_);
        inst_.shading_passes.shadow.render();
        DRW_draw_pass(page_copy_ps_);
      }
    }

    MEM_freeN(rect);
  }
  DRW_stats_group_end();

  DRW_view_set_active(view);
}

void ShadowModule::debug_draw(GPUFrameBuffer *view_fb, HiZBuffer &hiz)
{
  if (debug_draw_ps_ == nullptr) {
    return;
  }
  input_depth_tx_ = hiz.texture_get();

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
  DRW_draw_pass(surface_ps_);
}

/** \} */

}  // namespace blender::eevee
