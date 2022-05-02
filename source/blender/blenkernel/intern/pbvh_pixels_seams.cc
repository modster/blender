/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

#include "BKE_image.h"
#include "BKE_image_wrappers.hh"
#include "BKE_pbvh.h"
#include "BKE_pbvh_pixels.hh"

#include "IMB_imbuf_types.h"

#include "DNA_mesh_types.h"
#include "DNA_object_types.h"

#include "BLI_edgehash.h"
#include "BLI_vector.hh"

#include "pbvh_intern.h"

namespace blender::bke::pbvh::pixels {

/* Distance between a pixel and its edge that will be fixed. Value is in pixels space. */
constexpr float SEAMFIX_EDGE_DISTANCE = 3.5f;

struct EdgeLoop {
  /** Loop indexes that form an edge. */
  int l[2];
};

enum class EdgeCheckFlag {
  /** No connecting edge loop found. */
  Unconnected,
  /** A connecting edge loop found. */
  Connected,
};

struct EdgeCheck {
  EdgeCheckFlag flag;
  EdgeLoop first;
  EdgeLoop second;
  /* First vertex index of the first edge loop to determine winding order switching. */
  int first_v;
};

/** Do the two given EdgeLoops share the same uv coordinates. */
bool share_uv(const MLoopUV *ldata_uv, EdgeLoop &edge1, EdgeLoop &edge2)
{
  const float2 &uv_1_a = ldata_uv[edge1.l[0]].uv;
  const float2 &uv_1_b = ldata_uv[edge1.l[1]].uv;
  const float2 &uv_2_a = ldata_uv[edge2.l[0]].uv;
  const float2 &uv_2_b = ldata_uv[edge2.l[1]].uv;

  return (equals_v2v2(uv_1_a, uv_2_a) && equals_v2v2(uv_1_b, uv_2_b)) ||
         (equals_v2v2(uv_1_a, uv_2_b) && equals_v2v2(uv_1_b, uv_2_a));
}

/** Make a list of connected and unconnected edgeloops that require UV Seam fixes. */
void find_edges_that_need_fixing(const Mesh *mesh,
                                 const MLoopUV *ldata_uv,
                                 Vector<std::pair<EdgeLoop, EdgeLoop>> &r_connected,
                                 Vector<EdgeLoop> &r_unconnected)
{
  EdgeHash *eh = BLI_edgehash_new_ex(__func__, BLI_EDGEHASH_SIZE_GUESS_FROM_POLYS(mesh->totpoly));

  for (int p = 0; p < mesh->totpoly; p++) {
    MPoly &mpoly = mesh->mpoly[p];
    int prev_l = mpoly.loopstart + mpoly.totloop - 1;
    for (int l = 0; l < mpoly.totloop; l++) {
      MLoop &prev_mloop = mesh->mloop[prev_l];
      int current_l = mpoly.loopstart + l;
      MLoop &mloop = mesh->mloop[current_l];

      void **value_ptr;

      if (!BLI_edgehash_ensure_p(eh, prev_mloop.v, mloop.v, &value_ptr)) {
        EdgeCheck *value = MEM_cnew<EdgeCheck>(__func__);
        value->flag = EdgeCheckFlag::Unconnected;
        value->first.l[0] = prev_l;
        value->first.l[1] = current_l;
        value->first_v = prev_mloop.v;
        *value_ptr = value;
      }
      else {
        EdgeCheck *value = static_cast<EdgeCheck *>(*value_ptr);
        if (value->flag == EdgeCheckFlag::Unconnected) {
          value->flag = EdgeCheckFlag::Connected;
          /* Switch winding order to match the first edge. */
          if (prev_mloop.v == value->first_v) {
            value->second.l[0] = prev_l;
            value->second.l[1] = current_l;
          }
          else {
            value->second.l[0] = current_l;
            value->second.l[1] = prev_l;
          }
        }
      }

      prev_l = current_l;
    }
  }

  EdgeHashIterator iter;
  BLI_edgehashIterator_init(&iter, eh);
  while (!BLI_edgehashIterator_isDone(&iter)) {
    EdgeCheck *value = static_cast<EdgeCheck *>(BLI_edgehashIterator_getValue(&iter));
    switch (value->flag) {
      case EdgeCheckFlag::Unconnected: {
        r_unconnected.append(value->first);
        break;
      }
      case EdgeCheckFlag::Connected: {
        if (!share_uv(ldata_uv, value->first, value->second)) {
          r_connected.append(std::pair<EdgeLoop, EdgeLoop>(value->first, value->second));
          r_connected.append(std::pair<EdgeLoop, EdgeLoop>(value->second, value->first));
        }
        break;
      }
    }

    BLI_edgehashIterator_step(&iter);
  }

  BLI_edgehash_free(eh, MEM_freeN);
}

struct PixelInfo {
  const static uint32_t IS_EXTRACTED = 1 << 0;
  const static uint32_t IS_SEAM_FIX = 1 << 1;

  uint32_t node = 0;
  PixelInfo() = default;
  PixelInfo(const PixelInfo &other) = default;

  static PixelInfo from_node(int node_index)
  {
    PixelInfo result;
    result.node = node_index << 2 | PixelInfo::IS_EXTRACTED;
    return result;
  }

  static PixelInfo seam_fix()
  {
    PixelInfo result;
    result.node = IS_SEAM_FIX;
    return result;
  }

  uint32_t get_node_index() const
  {
    return node >> 2;
  }

  bool is_extracted() const
  {
    return (node & PixelInfo::IS_EXTRACTED) != 0;
  }
  bool is_seam_fix() const
  {
    return (node & PixelInfo::IS_SEAM_FIX) != 0;
  }
  bool is_empty_space() const
  {
    return !(is_extracted() || is_seam_fix());
  }
};

struct Bitmap {
  image::ImageTileWrapper image_tile;
  Vector<PixelInfo> bitmap;
  int2 resolution;

  Bitmap(image::ImageTileWrapper &image_tile, Vector<PixelInfo> bitmap, int2 resolution)
      : image_tile(image_tile), bitmap(bitmap), resolution(resolution)
  {
  }

  void mark_seam_fix(int2 image_coordinate)
  {
    int offset = image_coordinate.y * resolution.x + image_coordinate.x;
    bitmap[offset] = PixelInfo::seam_fix();
  }

  const PixelInfo &get_pixel_info(int2 image_coordinate) const
  {
    int offset = image_coordinate.y * resolution.x + image_coordinate.x;
    return bitmap[offset];
  }

  bool contains(const float2 &uv) const
  {
    int2 tile_offset = image_tile.get_tile_offset();
    float2 tile_uv(uv.x - tile_offset.x, uv.y - tile_offset.y);
    if (tile_uv.x < 0.0f || tile_uv.x >= 1.0f) {
      return false;
    }
    if (tile_uv.y < 0.0f || tile_uv.y >= 1.0f) {
      return false;
    }
    return true;
  }
};

struct Bitmaps {
  Vector<Bitmap> bitmaps;

  const Bitmap *find_containing_uv(float2 uv) const
  {
    for (const Bitmap &bitmap : bitmaps) {
      if (bitmap.contains(uv)) {
        return &bitmap;
      }
    }
    return nullptr;
  }
};

Vector<PixelInfo> create_tile_bitmap(const PBVH &pbvh,
                                     image::ImageTileWrapper &image_tile,
                                     ImBuf &image_buffer)
{
  Vector<PixelInfo> result(image_buffer.x * image_buffer.y);

  for (int n = 0; n < pbvh.totnode; n++) {
    PBVHNode *node = &pbvh.nodes[n];
    if ((node->flag & PBVH_Leaf) == 0) {
      continue;
    }
    NodeData *node_data = static_cast<NodeData *>(node->pixels.node_data);
    UDIMTilePixels *tile_node_data = node_data->find_tile_data(image_tile);
    if (tile_node_data == nullptr) {
      continue;
    }

    for (PackedPixelRow &pixel_row : tile_node_data->pixel_rows) {
      int pixel_offset = pixel_row.start_image_coordinate.y * image_buffer.x +
                         pixel_row.start_image_coordinate.x;
      for (int x = 0; x < pixel_row.num_pixels; x++) {
        result[pixel_offset] = PixelInfo::from_node(n);
        pixel_offset += 1;
      }
    }
  }
  return result;
}

Bitmaps create_tile_bitmap(const PBVH &pbvh, Image &image, ImageUser &image_user)
{
  Bitmaps result;
  ImageUser watertight = image_user;
  LISTBASE_FOREACH (ImageTile *, tile_data, &image.tiles) {
    image::ImageTileWrapper image_tile(tile_data);
    watertight.tile = image_tile.get_tile_number();
    ImBuf *image_buffer = BKE_image_acquire_ibuf(&image, &watertight, nullptr);
    if (image_buffer == nullptr) {
      continue;
    }

    Vector<PixelInfo> bitmap = create_tile_bitmap(pbvh, image_tile, *image_buffer);
    result.bitmaps.append(Bitmap(image_tile, bitmap, int2(image_buffer->x, image_buffer->y)));

    BKE_image_release_ibuf(&image, image_buffer, nullptr);
  }
  return result;
}

int2 find_source_pixel(const Bitmap &bitmap, float2 near_image_coord)
{
  const int SEARCH_RADIUS = 2;
  float min_distance = FLT_MAX;
  int2 result(0, 0);
  int2 image_coord(int(near_image_coord.x), int(near_image_coord.y));
  for (int v = image_coord.y - SEARCH_RADIUS; v <= image_coord.y + SEARCH_RADIUS; v++) {
    for (int u = image_coord.x - SEARCH_RADIUS; u <= image_coord.x + SEARCH_RADIUS; u++) {
      if (u < 0 || u >= bitmap.resolution.x || v < 0 || v >= bitmap.resolution.y) {
        /** Pixel not part of this tile. */
        continue;
      }

      int2 uv(u, v);
      const PixelInfo &pixel_info = bitmap.get_pixel_info(uv);
      if (!pixel_info.is_extracted()) {
        continue;
      }

      float distance = len_v2v2_int(uv, image_coord);
      if (distance < min_distance) {
        result = uv;
        min_distance = distance;
      }
    }
  }

  return result;
}

/** Clears all existing seam fixes in the given PBVH. */
static void pbvh_pixels_clear_seams(PBVH *pbvh)
{
  for (int n = 0; n < pbvh->totnode; n++) {
    PBVHNode &node = pbvh->nodes[n];
    if ((node.flag & PBVH_Leaf) == 0) {
      continue;
    }
    NodeData &node_data = BKE_pbvh_pixels_node_data_get(node);
    node_data.seams.clear();
  }
}

static void add_seam_fix(PBVHNode &node,
                         uint16_t src_tile_number,
                         int2 src_pixel,
                         uint16_t dst_tile_number,
                         int2 dst_pixel)
{
  NodeData &node_data = BKE_pbvh_pixels_node_data_get(node);
  UDIMSeamFixes &seam_fixes = node_data.ensure_seam_fixes(src_tile_number, dst_tile_number);
  seam_fixes.pixels.append(SeamFix{src_pixel, dst_pixel});
}

/* -------------------------------------------------------------------- */

/** \name Build fixes for connected edges.
 * \{ */

struct Projection {
  const Bitmap *bitmap;
  int2 pixel;
  PixelInfo pixel_info;
  bool is_valid;
};

/*
 * Project the point over onto the connected UV space. Taking into account the scale
 * difference.
 */
static void find_projection_source(const Bitmaps &bitmaps,
                                   const float distance_to_edge,
                                   const float lambda,
                                   const MLoopUV &uv1,
                                   const MLoopUV &uv2,
                                   const float scale_factor,
                                   Projection &r_projection)
{
  r_projection.is_valid = false;

  float2 closest_point;
  interp_v2_v2v2(closest_point, uv1.uv, uv2.uv, lambda);

  r_projection.bitmap = bitmaps.find_containing_uv(closest_point);
  if (r_projection.bitmap == nullptr) {
    return;
  }

  closest_point.x -= r_projection.bitmap->image_tile.get_tile_x_offset();
  closest_point.y -= r_projection.bitmap->image_tile.get_tile_y_offset();

  float2 direction;
  sub_v2_v2v2(direction, uv2.uv, uv1.uv);

  float2 perpedicular(direction.y, -direction.x);
  normalize_v2(perpedicular);
  perpedicular.x /= r_projection.bitmap->resolution.x;
  perpedicular.y /= r_projection.bitmap->resolution.y;
  float2 projected_coord = closest_point + perpedicular * distance_to_edge * scale_factor;
  projected_coord.x *= r_projection.bitmap->resolution.x;
  projected_coord.y *= r_projection.bitmap->resolution.y;
  r_projection.pixel = find_source_pixel(*r_projection.bitmap, projected_coord);
  r_projection.pixel_info = r_projection.bitmap->get_pixel_info(r_projection.pixel);
  r_projection.is_valid = r_projection.pixel_info.is_extracted();
}

static void build_fixes(PBVH &pbvh,
                        Bitmaps &bitmaps,
                        Bitmap &bitmap,
                        const rcti &uvbounds,
                        const MLoopUV &luv_a_1,
                        const MLoopUV &luv_a_2,
                        const MLoopUV &luv_b_1,
                        const MLoopUV &luv_b_2,
                        const float scale_factor)
{
  for (int v = uvbounds.ymin; v <= uvbounds.ymax; v++) {
    for (int u = uvbounds.xmin; u <= uvbounds.xmax; u++) {
      if (u < 0 || u >= bitmap.resolution.x || v < 0 || v >= bitmap.resolution.y) {
        /** Pixel not part of this tile. */
        continue;
      }

      int pixel_offset = v * bitmap.resolution.x + u;
      PixelInfo &pixel_info = bitmap.bitmap[pixel_offset];
      if (pixel_info.is_extracted() || pixel_info.is_seam_fix()) {
        /* Skip this pixel as it already has a solution. */
        continue;
      }

      float2 dst_uv_offset(bitmap.image_tile.get_tile_x_offset(),
                           bitmap.image_tile.get_tile_y_offset());
      float2 uv_coord(u, v);
      float2 uv(uv_coord.x / bitmap.resolution.x, uv_coord.y / bitmap.resolution.y);
      float2 closest_point;
      const float lambda = closest_to_line_v2(closest_point,
                                              uv,
                                              float2(luv_a_1.uv) - dst_uv_offset,
                                              float2(luv_a_2.uv) - dst_uv_offset);
      float2 closest_coord(closest_point.x * bitmap.resolution.x,
                           closest_point.y * bitmap.resolution.y);

      /* Distance to the edge in pixel space. */
      float distance_to_edge = len_v2v2(closest_coord, uv_coord);
      if (distance_to_edge > SEAMFIX_EDGE_DISTANCE) {
        continue;
      }

      Projection solution;
      find_projection_source(
          bitmaps, distance_to_edge, lambda, luv_b_1, luv_b_2, scale_factor, solution);
      if (!solution.is_valid) {
        find_projection_source(
            bitmaps, distance_to_edge, 1.0f - lambda, luv_b_2, luv_b_1, scale_factor, solution);
      }
      if (!solution.is_valid) {
        /* No solution found skip this pixel. */
        continue;
      }

      int2 destination_pixel(u, v);
      int src_node = solution.pixel_info.get_node_index();

      PBVHNode &node = pbvh.nodes[src_node];
      add_seam_fix(node,
                   solution.bitmap->image_tile.get_tile_number(),
                   solution.pixel,
                   bitmap.image_tile.get_tile_number(),
                   destination_pixel);
      bitmap.mark_seam_fix(destination_pixel);
    }
  }
}

static void build_fixes(PBVH &pbvh,
                        const Vector<std::pair<EdgeLoop, EdgeLoop>> &connected,
                        Bitmaps &bitmaps,
                        const MLoopUV *ldata_uv)
{
  for (const std::pair<EdgeLoop, EdgeLoop> &pair : connected) {
    // determine bounding rect in uv space + margin of 1;
    rctf uvbounds;
    BLI_rctf_init_minmax(&uvbounds);
    const MLoopUV &luv_a_1 = ldata_uv[pair.first.l[0]];
    const MLoopUV &luv_a_2 = ldata_uv[pair.first.l[1]];
    BLI_rctf_do_minmax_v(&uvbounds, luv_a_1.uv);
    BLI_rctf_do_minmax_v(&uvbounds, luv_a_2.uv);

    const MLoopUV &luv_b_1 = ldata_uv[pair.second.l[0]];
    const MLoopUV &luv_b_2 = ldata_uv[pair.second.l[1]];

    const float scale_factor = len_v2v2(luv_b_1.uv, luv_b_2.uv) / len_v2v2(luv_a_1.uv, luv_a_2.uv);

    for (Bitmap &bitmap : bitmaps.bitmaps) {
      rcti uvbounds_i;
      const int MARGIN = 2;
      uvbounds_i.xmin = (uvbounds.xmin - bitmap.image_tile.get_tile_x_offset()) *
                            bitmap.resolution[0] -
                        MARGIN;
      uvbounds_i.ymin = (uvbounds.ymin - bitmap.image_tile.get_tile_y_offset()) *
                            bitmap.resolution[1] -
                        MARGIN;
      uvbounds_i.xmax = (uvbounds.xmax - bitmap.image_tile.get_tile_x_offset()) *
                            bitmap.resolution[0] +
                        MARGIN;
      uvbounds_i.ymax = (uvbounds.ymax - bitmap.image_tile.get_tile_y_offset()) *
                            bitmap.resolution[1] +
                        MARGIN;

      build_fixes(
          pbvh, bitmaps, bitmap, uvbounds_i, luv_a_1, luv_a_2, luv_b_1, luv_b_2, scale_factor);
    }
  }
}

/** \} */

/* -------------------------------------------------------------------- */

/** \name Build fixes for unconnected edges.
 * \{ */

static void build_fixes(
    PBVH &pbvh, Bitmap &bitmap, const rcti &uvbounds, const MLoopUV &luv_1, const MLoopUV &luv_2)
{
  for (int v = uvbounds.ymin; v <= uvbounds.ymax; v++) {
    for (int u = uvbounds.xmin; u <= uvbounds.xmax; u++) {
      if (u < 0 || u >= bitmap.resolution[0] || v < 0 || v >= bitmap.resolution[1]) {
        /** Pixel not part of this tile. */
        continue;
      }
      int pixel_offset = v * bitmap.resolution[0] + u;
      PixelInfo &pixel_info = bitmap.bitmap[pixel_offset];
      if (pixel_info.is_extracted() || pixel_info.is_seam_fix()) {
        /* Skip this pixel as it already has a solution. */
        continue;
      }

      float2 dst_uv_offset(bitmap.image_tile.get_tile_x_offset(),
                           bitmap.image_tile.get_tile_y_offset());
      float2 uv(float(u) / bitmap.resolution[0], float(v) / bitmap.resolution[1]);
      float2 closest_point;
      closest_to_line_v2(
          closest_point, uv, float2(luv_1.uv) - dst_uv_offset, float2(luv_2.uv) - dst_uv_offset);

      /* Calculate the distance in pixel space. */
      float2 uv_coord(u, v);
      float2 closest_coord(closest_point.x * bitmap.resolution.x,
                           closest_point.y * bitmap.resolution.y);
      float distance_to_edge = len_v2v2(uv_coord, closest_coord);
      if (distance_to_edge > SEAMFIX_EDGE_DISTANCE) {
        continue;
      }

      int2 source_pixel = find_source_pixel(bitmap, closest_coord);
      int2 destination_pixel(u, v);

      PixelInfo src_pixel_info = bitmap.get_pixel_info(source_pixel);
      if (!src_pixel_info.is_extracted()) {
        continue;
      }
      int src_node = src_pixel_info.get_node_index();

      PBVHNode &node = pbvh.nodes[src_node];
      add_seam_fix(node,
                   bitmap.image_tile.get_tile_number(),
                   source_pixel,
                   bitmap.image_tile.get_tile_number(),
                   destination_pixel);
      bitmap.mark_seam_fix(destination_pixel);
    }
  }
}

static void build_fixes(PBVH &pbvh,
                        const Vector<EdgeLoop> &unconnected,
                        Bitmaps &bitmaps,
                        const MLoopUV *ldata_uv)
{
  for (const EdgeLoop &unconnected_loop : unconnected) {
    // determine bounding rect in uv space + margin of 1;
    rctf uvbounds;
    BLI_rctf_init_minmax(&uvbounds);
    const MLoopUV &luv_1 = ldata_uv[unconnected_loop.l[0]];
    const MLoopUV &luv_2 = ldata_uv[unconnected_loop.l[1]];
    BLI_rctf_do_minmax_v(&uvbounds, luv_1.uv);
    BLI_rctf_do_minmax_v(&uvbounds, luv_2.uv);

    for (Bitmap &bitmap : bitmaps.bitmaps) {
      rcti uvbounds_i;
      const int MARGIN = 1;
      uvbounds_i.xmin = (uvbounds.xmin - bitmap.image_tile.get_tile_x_offset()) *
                            bitmap.resolution[0] -
                        MARGIN;
      uvbounds_i.ymin = (uvbounds.ymin - bitmap.image_tile.get_tile_y_offset()) *
                            bitmap.resolution[1] -
                        MARGIN;
      uvbounds_i.xmax = (uvbounds.xmax - bitmap.image_tile.get_tile_x_offset()) *
                            bitmap.resolution[0] +
                        MARGIN;
      uvbounds_i.ymax = (uvbounds.ymax - bitmap.image_tile.get_tile_y_offset()) *
                            bitmap.resolution[1] +
                        MARGIN;

      build_fixes(pbvh, bitmap, uvbounds_i, luv_1, luv_2);
    }
  }
}

/** \} */

void BKE_pbvh_pixels_rebuild_seams(
    PBVH *pbvh, const Mesh *mesh, Image *image, ImageUser *image_user, const MLoopUV *ldata_uv)
{

  // find seams.
  // for each edge
  Vector<std::pair<EdgeLoop, EdgeLoop>> connected;
  Vector<EdgeLoop> unconnected;
  find_edges_that_need_fixing(mesh, ldata_uv, connected, unconnected);

  // Make a bitmap per tile indicating pixels that have already been assigned to a PBVHNode.
  Bitmaps bitmaps = create_tile_bitmap(*pbvh, *image, *image_user);

  pbvh_pixels_clear_seams(pbvh);
  /* Fix connected edges before unconnected to improve quality. */
  build_fixes(*pbvh, connected, bitmaps, ldata_uv);
  build_fixes(*pbvh, unconnected, bitmaps, ldata_uv);
}

void BKE_pbvh_pixels_fix_seams(PBVHNode *node, Image *image, ImageUser *image_user)
{
  NodeData &node_data = BKE_pbvh_pixels_node_data_get(*node);
  ImageUser iuser = *image_user;

  for (UDIMSeamFixes &fixes : node_data.seams) {
    iuser.tile = fixes.dst_tile_number;
    ImBuf *dst_image_buffer = BKE_image_acquire_ibuf(image, &iuser, nullptr);
    if (dst_image_buffer == nullptr) {
      continue;
    }

    iuser.tile = fixes.src_tile_number;
    ImBuf *src_image_buffer = BKE_image_acquire_ibuf(image, &iuser, nullptr);
    if (src_image_buffer == nullptr) {
      continue;
    }

    /** Determine the region to update by checking actual changes. */
    rcti region_to_update;
    BLI_rcti_init_minmax(&region_to_update);

    if (src_image_buffer->rect_float != nullptr && dst_image_buffer->rect_float != nullptr) {
      for (SeamFix &fix : fixes.pixels) {
        int src_offset = fix.src_pixel.y * src_image_buffer->x + fix.src_pixel.x;
        int dst_offset = fix.dst_pixel.y * dst_image_buffer->x + fix.dst_pixel.x;

        if (equals_v4v4(&dst_image_buffer->rect_float[dst_offset * 4],
                        &src_image_buffer->rect_float[src_offset * 4])) {
          continue;
        }
        BLI_rcti_do_minmax_v(&region_to_update, fix.dst_pixel);
        copy_v4_v4(&dst_image_buffer->rect_float[dst_offset * 4],
                   &src_image_buffer->rect_float[src_offset * 4]);
      }
    }
    else if (src_image_buffer->rect != nullptr && dst_image_buffer->rect != nullptr) {
      for (SeamFix &fix : fixes.pixels) {
        int src_offset = fix.src_pixel.y * src_image_buffer->x + fix.src_pixel.x;
        int dst_offset = fix.dst_pixel.y * dst_image_buffer->x + fix.dst_pixel.x;
        if (dst_image_buffer->rect[dst_offset] == src_image_buffer->rect[src_offset]) {
          continue;
        }
        BLI_rcti_do_minmax_v(&region_to_update, fix.dst_pixel);
        dst_image_buffer->rect[dst_offset] = src_image_buffer->rect[src_offset];
      }
    }

    /* Mark dst_image_buffer region dirty covering each dst_pixel. */
    if (BLI_rcti_is_valid(&region_to_update)) {
      LISTBASE_FOREACH (ImageTile *, image_tile, &image->tiles) {
        if (image_tile->tile_number != fixes.dst_tile_number) {
          continue;
        }

        BKE_image_partial_update_mark_region(
            image, image_tile, dst_image_buffer, &region_to_update);
        break;
      }
    }
    BKE_image_release_ibuf(image, src_image_buffer, nullptr);
    BKE_image_release_ibuf(image, dst_image_buffer, nullptr);
  }
}

}  // namespace blender::bke::pbvh::pixels
