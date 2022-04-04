/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

#include "BKE_customdata.h"
#include "BKE_mesh_mapping.h"
#include "BKE_pbvh.h"
#include "BKE_pbvh.hh"

#include "DNA_image_types.h"
#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"
#include "DNA_object_types.h"

#include "BLI_math.h"
#include "BLI_task.h"

#include "BKE_image_wrappers.hh"

#include "bmesh.h"

#include "pbvh_intern.h"

namespace blender::bke::pbvh::pixels::extractor {

/**
 * Keep track of visited polygons.
 *
 * Uses a std::vector<bool> to reduce memory requirements.
 * TODO(jbakker): Should be replaced by BLI bool vector when available.
 */
class VisitedPolygons : std::vector<bool> {
 public:
  VisitedPolygons(int64_t polygon_len) : std::vector<bool>(polygon_len)
  {
  }

  /**
   * Check of the given poly_index has already been visited.
   * Marks the given polygon as visited and return the previous visited state.
   */
  bool tag_visited(const int poly_index)
  {
    bool visited = (*this)[poly_index];
    (*this)[poly_index] = true;
    return visited;
  }
};

/**
 * @brief Calculate the delta of two neighbour uv coordinates in the given image buffer.
 */
static float3 calc_barycentric_delta(const ImBuf *image_buffer,
                                     const float2 uvs[3],
                                     const int x,
                                     const int y)
{
  const float2 start_uv(float(x) / image_buffer->x, float(y) / image_buffer->y);
  const float2 end_uv(float(x + 1) / image_buffer->x, float(y) / image_buffer->y);
  const BarycentricWeights start_barycentric(uvs[0], uvs[1], uvs[2], start_uv);
  const BarycentricWeights end_barycentric(uvs[0], uvs[1], uvs[2], end_uv);
  const float3 delta_barycentric = end_barycentric - start_barycentric;
  return delta_barycentric;
}

static void extract_barycentric_pixels(TileData &tile_data,
                                       const ImBuf *image_buffer,
                                       const int triangle_index,
                                       const float2 uvs[3],
                                       const int minx,
                                       const int miny,
                                       const int maxx,
                                       const int maxy)
{
  for (int y = miny; y < maxy; y++) {
    bool start_detected = false;
    PixelsPackage package;
    package.triangle_index = triangle_index;
    package.num_pixels = 0;
    int x;

    for (x = minx; x < maxx; x++) {
      float2 uv(float(x) / image_buffer->x, float(y) / image_buffer->y);
      const BarycentricWeights barycentric(uvs[0], uvs[1], uvs[2], uv);
      const bool is_inside = barycentric.is_inside_triangle();
      if (!start_detected && is_inside) {
        start_detected = true;
        package.start_image_coordinate = ushort2(x, y);
        package.start_barycentric_coord = barycentric;
      }
      else if (start_detected && !is_inside) {
        break;
      }
    }

    if (!start_detected) {
      continue;
    }
    package.num_pixels = x - package.start_image_coordinate.x;
    tile_data.encoded_pixels.append(package);
  }
}

static void init_triangles(PBVH *pbvh,
                           PBVHNode *node,
                           NodeData *node_data,
                           const MeshElemMap *pmap,
                           const MPoly *mpoly,
                           const MLoop *mloop,
                           VisitedPolygons &visited_polygons)
{
  PBVHVertexIter vd;

  BKE_pbvh_vertex_iter_begin (pbvh, node, vd, PBVH_ITER_UNIQUE) {
    const MeshElemMap *vert_map = &pmap[vd.index];
    for (int j = 0; j < pmap[vd.index].count; j++) {
      const int poly_index = vert_map->indices[j];
      if (visited_polygons.tag_visited(poly_index)) {
        continue;
      }

      const MPoly *p = &mpoly[poly_index];
      const MLoop *loopstart = &mloop[p->loopstart];
      for (int l = 0; l < p->totloop - 2; l++) {
        Triangle triangle;
        triangle.loop_indices = int3(p->loopstart, p->loopstart + l + 1, p->loopstart + l + 2);
        triangle.vert_indices = int3(loopstart[0].v, loopstart[l + 1].v, loopstart[l + 2].v);
        triangle.poly_index = poly_index;
        node_data->triangles.append(triangle);
      }
    }
  }
  BKE_pbvh_vertex_iter_end;
}

struct EncodePixelsUserData {
  Image *image;
  ImageUser *image_user;
  Vector<PBVHNode *> *nodes;
  MLoopUV *ldata_uv;
};

static void do_encode_pixels(void *__restrict userdata,
                             const int n,
                             const TaskParallelTLS *__restrict UNUSED(tls))
{
  EncodePixelsUserData *data = static_cast<EncodePixelsUserData *>(userdata);
  Image *image = data->image;
  ImageUser image_user = *data->image_user;

  PBVHNode *node = (*data->nodes)[n];
  NodeData *node_data = static_cast<NodeData *>(node->pixels.node_data);
  LISTBASE_FOREACH (ImageTile *, tile, &data->image->tiles) {
    image::ImageTileWrapper image_tile(tile);
    image_user.tile = image_tile.get_tile_number();
    ImBuf *image_buffer = BKE_image_acquire_ibuf(image, &image_user, nullptr);
    if (image_buffer == nullptr) {
      continue;
    }

    float2 tile_offset = float2(image_tile.get_tile_offset());
    TileData tile_data;

    Triangles &triangles = node_data->triangles;
    for (int triangle_index = 0; triangle_index < triangles.size(); triangle_index++) {
      int3 loop_indices = triangles.get_loop_indices(triangle_index);
      float2 uvs[3] = {
          float2(data->ldata_uv[loop_indices[0]].uv) - tile_offset,
          float2(data->ldata_uv[loop_indices[1]].uv) - tile_offset,
          float2(data->ldata_uv[loop_indices[2]].uv) - tile_offset,
      };

      const float minv = clamp_f(min_fff(uvs[0].y, uvs[1].y, uvs[2].y), 0.0f, 1.0f);
      const int miny = floor(minv * image_buffer->y);
      const float maxv = clamp_f(max_fff(uvs[0].y, uvs[1].y, uvs[2].y), 0.0f, 1.0f);
      const int maxy = min_ii(ceil(maxv * image_buffer->y), image_buffer->y);
      const float minu = clamp_f(min_fff(uvs[0].x, uvs[1].x, uvs[2].x), 0.0f, 1.0f);
      const int minx = floor(minu * image_buffer->x);
      const float maxu = clamp_f(max_fff(uvs[0].x, uvs[1].x, uvs[2].x), 0.0f, 1.0f);
      const int maxx = min_ii(ceil(maxu * image_buffer->x), image_buffer->x);

      TrianglePaintInput &triangle = triangles.get_paint_input(triangle_index);
      triangle.add_barycentric_coord_x = calc_barycentric_delta(image_buffer, uvs, minx, miny);
      extract_barycentric_pixels(
          tile_data, image_buffer, triangle_index, uvs, minx, miny, maxx, maxy);
    }

    BKE_image_release_ibuf(image, image_buffer, nullptr);

    if (tile_data.encoded_pixels.is_empty()) {
      continue;
    }

    tile_data.tile_number = image_tile.get_tile_number();
    node_data->tiles.append(tile_data);
  }
  node_data->triangles.cleanup_after_init();
  node->flag = static_cast<PBVHNodeFlags>(node->flag & ~PBVH_UpdatePixels);
}

static bool should_pixels_be_updated(PBVHNode *node)
{
  if ((node->flag & PBVH_Leaf) == 0) {
    return false;
  }
  if ((node->flag & PBVH_UpdatePixels) != 0) {
    return true;
  }
  NodeData *node_data = static_cast<NodeData *>(node->pixels.node_data);
  if (node_data != nullptr) {
    return false;
  }
  return true;
}

/**
 * Does this the given node contains a list with owning polygons.
 *
 * The owning polygons are stored per triangle inside the node.
 */
static bool contains_polygons(PBVHNode *node)
{
  if ((node->flag & PBVH_Leaf) == 0) {
    return false;
  }
  NodeData *node_data = static_cast<NodeData *>(node->pixels.node_data);
  if (node_data == nullptr) {
    return false;
  }
  return true;
}

static int64_t count_nodes_to_update(PBVH *pbvh)
{
  int64_t result = 0;
  for (int n = 0; n < pbvh->totnode; n++) {
    PBVHNode *node = &pbvh->nodes[n];
    if (should_pixels_be_updated(node)) {
      result++;
    }
  }
  return result;
}

/**
 * Find the nodes that needs to be updated.
 *
 * The nodes that require updated are added to the r_nodes_to_update parameter.
 * Will fill in r_visited_polygons with polygons that are owned by nodes that do not require
 * updates.
 *
 * returns if there were any nodes found (true).
 */
static bool find_nodes_to_update(PBVH *pbvh,
                                 Vector<PBVHNode *> &r_nodes_to_update,
                                 VisitedPolygons &r_visited_polygons)
{
  int64_t nodes_to_update_len = count_nodes_to_update(pbvh);
  if (nodes_to_update_len == 0) {
    return false;
  }

  r_nodes_to_update.reserve(nodes_to_update_len);

  for (int n = 0; n < pbvh->totnode; n++) {
    PBVHNode *node = &pbvh->nodes[n];
    if (should_pixels_be_updated(node)) {
      r_nodes_to_update.append(node);

      if (node->pixels.node_data == nullptr) {
        NodeData *node_data = MEM_new<NodeData>(__func__);
        node->pixels.node_data = node_data;
      }
      else {
        NodeData *node_data = static_cast<NodeData *>(node->pixels.node_data);
        node_data->clear_data();
      }
    }
    else if (contains_polygons(node)) {
      /* Mark polygons that are owned by other leafs, so they don't be added to new other PBVH_Leaf
       * nodes. */
      Triangles &triangles = BKE_pbvh_pixels_triangles_get(*node);
      for (int &poly_index : triangles.poly_indices) {
        r_visited_polygons.tag_visited(poly_index);
      }
    }
  }

  return true;
}

static void update_pixels(PBVH *pbvh,
                          const MeshElemMap *pmap,
                          const struct MPoly *mpoly,
                          const struct MLoop *mloop,
                          struct CustomData *ldata,
                          int tot_poly,
                          struct Image *image,
                          struct ImageUser *image_user)
{
  Vector<PBVHNode *> nodes_to_update;
  VisitedPolygons visited_polygons(tot_poly);

  if (!find_nodes_to_update(pbvh, nodes_to_update, visited_polygons)) {
    return;
  }

  MLoopUV *ldata_uv = static_cast<MLoopUV *>(CustomData_get_layer(ldata, CD_MLOOPUV));
  if (ldata_uv == nullptr) {
    return;
  }

  for (PBVHNode *node : nodes_to_update) {
    NodeData *node_data = static_cast<NodeData *>(node->pixels.node_data);
    init_triangles(pbvh, node, node_data, pmap, mpoly, mloop, visited_polygons);
  }

  EncodePixelsUserData user_data;
  user_data.image = image;
  user_data.image_user = image_user;
  user_data.ldata_uv = ldata_uv;
  user_data.nodes = &nodes_to_update;

  TaskParallelSettings settings;
  BKE_pbvh_parallel_range_settings(&settings, true, nodes_to_update.size());
  BLI_task_parallel_range(0, nodes_to_update.size(), &user_data, do_encode_pixels, &settings);

//#define DO_PRINT_STATISTICS
#ifdef DO_PRINT_STATISTICS
  /* Print some statistics about compression ratio. */
  {
    int64_t compressed_data_len = 0;
    int64_t num_pixels = 0;
    for (int n = 0; n < pbvh->totnode; n++) {
      PBVHNode *node = &pbvh->nodes[n];
      if ((node->flag & PBVH_Leaf) == 0) {
        continue;
      }
      NodeData *node_data = static_cast<NodeData *>(node->pixels.node_data);
      compressed_data_len += node_data->triangles.mem_size();
      for (const TileData &tile_data : node_data->tiles) {
        compressed_data_len += tile_data.encoded_pixels.size() * sizeof(PixelsPackage);
        for (const PixelsPackage &encoded_pixels : tile_data.encoded_pixels) {
          num_pixels += encoded_pixels.num_pixels;
        }
      }
    }
    printf("Encoded %lld pixels in %lld bytes (%f bytes per pixel)\n",
           num_pixels,
           compressed_data_len,
           float(compressed_data_len) / num_pixels);
  }
#endif

//#define DO_WATERTIGHT_CHECK
#ifdef DO_WATERTIGHT_CHECK
  for (int n = 0; n < nodes_to_initialize.size(); n++) {
    PBVHNode *node = nodes[n];
    NodeData *node_data = static_cast<NodeData *>(BKE_pbvh_node_texture_paint_data_get(node));
    for (PixelsPackage &encoded_pixels : node_data->encoded_pixels) {
      int pixel_offset = encoded_pixels.start_image_coordinate.y * image_buffer->x +
                         encoded_pixels.start_image_coordinate.x;
      for (int x = 0; x < encoded_pixels.num_pixels; x++) {
        copy_v4_fl(&image_buffer->rect_float[pixel_offset * 4], 1.0);
        pixel_offset += 1;
      }
    }
  }

#endif
}

}  // namespace blender::bke::pbvh::pixels::extractor

namespace blender::bke::pbvh::pixels {

Triangles &BKE_pbvh_pixels_triangles_get(PBVHNode &node)
{
  BLI_assert(node.pixels.node_data != nullptr);
  NodeData *node_data = static_cast<NodeData *>(node.pixels.node_data);
  return node_data->triangles;
}

TileData *BKE_pbvh_pixels_tile_data_get(PBVHNode &node, const image::ImageTileWrapper &image_tile)
{
  BLI_assert(node.pixels.node_data != nullptr);
  NodeData *node_data = static_cast<NodeData *>(node.pixels.node_data);
  return node_data->find_tile_data(image_tile);
}

void BKE_pbvh_pixels_mark_dirty(PBVHNode &node)
{
  BLI_assert(node.pixels.node_data != nullptr);
  NodeData *node_data = static_cast<NodeData *>(node.pixels.node_data);
  node_data->flags.dirty |= true;
}

void BKE_pbvh_pixels_mark_image_dirty(PBVHNode &node, Image &image, ImageUser &image_user)
{
  BLI_assert(node.pixels.node_data != nullptr);
  NodeData *node_data = static_cast<NodeData *>(node.pixels.node_data);
  if (node_data->flags.dirty) {
    ImageUser local_image_user = image_user;
    void *image_lock;
    LISTBASE_FOREACH (ImageTile *, tile, &image.tiles) {
      image::ImageTileWrapper image_tile(tile);
      local_image_user.tile = image_tile.get_tile_number();
      ImBuf *image_buffer = BKE_image_acquire_ibuf(&image, &local_image_user, &image_lock);
      if (image_buffer == nullptr) {
        continue;
      }

      node_data->mark_region(image, image_tile, *image_buffer);
      BKE_image_release_ibuf(&image, image_buffer, image_lock);
    }
    node_data->flags.dirty = false;
  }
}

}  // namespace blender::bke::pbvh::pixels

extern "C" {
using namespace blender::bke::pbvh::pixels::extractor;
using namespace blender::bke::pbvh::pixels;

void BKE_pbvh_build_pixels(PBVH *pbvh,
                           const struct MeshElemMap *pmap,
                           const struct MPoly *mpoly,
                           const struct MLoop *mloop,
                           struct CustomData *ldata,
                           int tot_poly,
                           struct Image *image,
                           struct ImageUser *image_user)
{
  update_pixels(pbvh, pmap, mpoly, mloop, ldata, tot_poly, image, image_user);
}

void pbvh_pixels_free(PBVHNode *node)
{
  NodeData *node_data = static_cast<NodeData *>(node->pixels.node_data);
  MEM_delete(node_data);
  node->pixels.node_data = nullptr;
}
}
