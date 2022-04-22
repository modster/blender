/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

#include "BKE_image.h"
#include "BKE_image_wrappers.hh"
#include "BKE_pbvh.h"
#include "BKE_pbvh_pixels.hh"

#include "IMB_imbuf_types.h"

#include "DNA_mesh_types.h"
#include "DNA_object_types.h"

#include "BLI_vector.hh"

#include "bmesh.h"

#include "pbvh_intern.h"

using BMLoopConnection = std::pair<BMLoop *, BMLoop *>;

namespace blender::bke::pbvh::pixels {

/**
 * Find loops that are connected in 3d space, but not in uv space. Or loops that don't have any
 * connection at all.
 *
 * TODO better name would be to find loops that need uv seam fixes.
 */
void find_connected_loops(BMesh *bm,
                          const int cd_loop_uv_offset,
                          Vector<BMLoopConnection> &r_connected,
                          Vector<BMLoop *> &r_unconnected)
{
  BMEdge *e;
  BMIter eiter;
  BMLoop *l;
  BMIter liter;
  BM_ITER_MESH (e, &eiter, bm, BM_EDGES_OF_MESH) {
    bool first = true;
    bool connection_found = false;
    BMLoop *l_first;

    BM_ITER_ELEM (l, &liter, e, BM_LOOPS_OF_EDGE) {
      if (first) {
        l_first = l;
        first = false;
      }
      else {
        connection_found = true;
        if (!BM_loop_uv_share_edge_check(l_first, l, cd_loop_uv_offset)) {
          /* Edge detected that is connected in 3d space, but not in uv space. */
          r_connected.append(BMLoopConnection(l_first, l));
          r_connected.append(BMLoopConnection(l, l_first));
          break;
        }
      }
    }
    if (!connection_found) {
      BLI_assert(!first);
      r_unconnected.append(l_first);
    }
  }
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
};

struct Bitmaps {
  Vector<Bitmap> bitmaps;
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

int2 find_source_pixel(Bitmap &bitmap, float2 near_image_coord)
{
  // TODO: We should take lambda into account.
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

static void build_fixes(PBVH &pbvh,
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
      float2 uv_coord(u, v);
      // What is the distance to the edge.
      float2 uv(uv_coord.x / bitmap.resolution.x, uv_coord.y / bitmap.resolution.y);
      float2 closest_point;
      // TODO: Should we use lambda to reduce artifacts?
      const float lambda = closest_to_line_v2(closest_point, uv, luv_a_1.uv, luv_a_2.uv);
      float2 closest_coord(closest_point.x * bitmap.resolution.x,
                           closest_point.y * bitmap.resolution.y);

      /* Distance to the edge in pixel space. */
      float distance_to_edge = len_v2v2(closest_coord, uv_coord);
      if (distance_to_edge > 3.0f) {
        continue;
      }

      /*
       * Project the point over onto the connected UV space. Taking into account the scale
       * difference.
       */
      float2 other_closest_point;
      interp_v2_v2v2(other_closest_point, luv_b_2.uv, luv_b_1.uv, lambda);
      float2 direction_b;
      sub_v2_v2v2(direction_b, luv_b_2.uv, luv_b_1.uv);
      float2 perpedicular_b(direction_b.y, -direction_b.x);
      normalize_v2(perpedicular_b);
      perpedicular_b.x /= bitmap.resolution.x;
      perpedicular_b.y /= bitmap.resolution.y;
      float2 projected_coord_a = other_closest_point +
                                 perpedicular_b * distance_to_edge * scale_factor;
      projected_coord_a.x *= bitmap.resolution.x;
      projected_coord_a.y *= bitmap.resolution.y;
      int2 source_pixel = find_source_pixel(bitmap, projected_coord_a);
      PixelInfo src_pixel_info = bitmap.get_pixel_info(source_pixel);

      if (!src_pixel_info.is_extracted()) {
        float2 projected_coord_b = other_closest_point -
                                   perpedicular_b * distance_to_edge * scale_factor;
        projected_coord_b.x *= bitmap.resolution.x;
        projected_coord_b.y *= bitmap.resolution.y;
        source_pixel = find_source_pixel(bitmap, projected_coord_b);
        src_pixel_info = bitmap.get_pixel_info(source_pixel);
      }

      if (!src_pixel_info.is_extracted()) {
        continue;
      }

      int2 destination_pixel(u, v);
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
                        const Vector<BMLoopConnection> &connected,
                        Bitmaps &bitmaps,
                        const int cd_loop_uv_offset)
{
  for (const BMLoopConnection &pair : connected) {
    // determine bounding rect in uv space + margin of 1;
    rctf uvbounds;
    BLI_rctf_init_minmax(&uvbounds);
    MLoopUV *luv_a_1 = static_cast<MLoopUV *>(
        BM_ELEM_CD_GET_VOID_P(pair.first, cd_loop_uv_offset));
    MLoopUV *luv_a_2 = static_cast<MLoopUV *>(
        BM_ELEM_CD_GET_VOID_P(pair.first->next, cd_loop_uv_offset));
    BLI_rctf_do_minmax_v(&uvbounds, luv_a_1->uv);
    BLI_rctf_do_minmax_v(&uvbounds, luv_a_2->uv);

    MLoopUV *luv_b_1 = static_cast<MLoopUV *>(
        BM_ELEM_CD_GET_VOID_P(pair.second, cd_loop_uv_offset));
    MLoopUV *luv_b_2 = static_cast<MLoopUV *>(
        BM_ELEM_CD_GET_VOID_P(pair.second->next, cd_loop_uv_offset));

    const float scale_factor = len_v2v2(luv_b_1->uv, luv_b_2->uv) /
                               len_v2v2(luv_a_1->uv, luv_a_2->uv);

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

      build_fixes(pbvh, bitmap, uvbounds_i, *luv_a_1, *luv_a_2, *luv_b_1, *luv_b_2, scale_factor);
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

      // What is the distance to the edge.
      float2 uv(float(u) / bitmap.resolution[0], float(v) / bitmap.resolution[1]);
      float2 closest_point;
      // TODO: Should we use lambda to reduce artifacts?
      closest_to_line_v2(closest_point, uv, luv_1.uv, luv_2.uv);

      /* Calculate the distance in pixel space. */
      float2 uv_coord(u, v);
      float2 closest_coord(closest_point.x * bitmap.resolution.x,
                           closest_point.y * bitmap.resolution.y);
      float distance_to_edge = len_v2v2(uv_coord, closest_coord);
      if (distance_to_edge > 3.0f) {
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
                        const Vector<BMLoop *> &unconnected,
                        Bitmaps &bitmaps,
                        const int cd_loop_uv_offset)
{
  for (const BMLoop *unconnected_loop : unconnected) {
    // determine bounding rect in uv space + margin of 1;
    rctf uvbounds;
    BLI_rctf_init_minmax(&uvbounds);
    MLoopUV *luv_1 = static_cast<MLoopUV *>(
        BM_ELEM_CD_GET_VOID_P(unconnected_loop, cd_loop_uv_offset));
    MLoopUV *luv_2 = static_cast<MLoopUV *>(
        BM_ELEM_CD_GET_VOID_P(unconnected_loop->next, cd_loop_uv_offset));
    BLI_rctf_do_minmax_v(&uvbounds, luv_1->uv);
    BLI_rctf_do_minmax_v(&uvbounds, luv_2->uv);

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

      build_fixes(pbvh, bitmap, uvbounds_i, *luv_1, *luv_2);
    }
  }
}

/** \} */

void BKE_pbvh_pixels_rebuild_seams(
    PBVH *pbvh, Mesh *mesh, Image *image, ImageUser *image_user, int cd_loop_uv_offset)
{
  const BMAllocTemplate allocsize = BMALLOC_TEMPLATE_FROM_ME(mesh);

  BMeshCreateParams bmesh_create_params{};
  bmesh_create_params.use_toolflags = true;
  BMesh *bm = BM_mesh_create(&allocsize, &bmesh_create_params);

  BMeshFromMeshParams from_mesh_params{};
  from_mesh_params.calc_face_normal = false;
  from_mesh_params.calc_vert_normal = false;
  BM_mesh_bm_from_me(bm, mesh, &from_mesh_params);

  // find seams.
  // for each edge
  Vector<BMLoopConnection> connected;
  Vector<BMLoop *> unconnected;
  find_connected_loops(bm, cd_loop_uv_offset, connected, unconnected);

  // Make a bitmap per tile indicating pixels that have already been assigned to a PBVHNode.
  Bitmaps bitmaps = create_tile_bitmap(*pbvh, *image, *image_user);

  pbvh_pixels_clear_seams(pbvh);
  /* Fix connected edges before unconnected to improve quality. */
  build_fixes(*pbvh, connected, bitmaps, cd_loop_uv_offset);
  build_fixes(*pbvh, unconnected, bitmaps, cd_loop_uv_offset);

  BM_mesh_free(bm);
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
