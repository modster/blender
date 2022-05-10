/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

#include "BKE_customdata.h"
#include "BKE_mesh_mapping.h"
#include "BKE_pbvh.h"
#include "BKE_pbvh_pixels.hh"

#include "DNA_image_types.h"
#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"
#include "DNA_object_types.h"

#include "BLI_math.h"
#include "BLI_task.h"
#include "PIL_time.h"

#include "BKE_global.h"
#include "BKE_image_wrappers.hh"

#include "bmesh.h"

#include "pbvh_intern.h"

#include <atomic>

namespace blender::bke::pbvh::pixels {

/**
 * Calculate the delta of two neighbor UV coordinates in the given image buffer.
 */
static float2 calc_barycentric_delta(const float2 uvs[3],
                                     const float2 start_uv,
                                     const float2 end_uv)
{

  float3 start_barycentric;
  barycentric_weights_v2(uvs[0], uvs[1], uvs[2], start_uv, start_barycentric);
  float3 end_barycentric;
  barycentric_weights_v2(uvs[0], uvs[1], uvs[2], end_uv, end_barycentric);
  float3 barycentric = end_barycentric - start_barycentric;
  return float2(barycentric.x, barycentric.y);
}

static float2 calc_barycentric_delta_x(const ImBuf *image_buffer,
                                       const float2 uvs[3],
                                       const int x,
                                       const int y)
{
  const float2 start_uv(float(x) / image_buffer->x, float(y) / image_buffer->y);
  const float2 end_uv(float(x + 1) / image_buffer->x, float(y) / image_buffer->y);
  return calc_barycentric_delta(uvs, start_uv, end_uv);
}

int count_node_pixels(PBVHNode &node)
{
  if (!node.pixels.node_data) {
    return 0;
  }

  NodeData &data = BKE_pbvh_pixels_node_data_get(node);

  int totpixel = 0;

  for (UDIMTilePixels &tile : data.tiles) {
    for (PackedPixelRow &row : tile.pixel_rows) {
      totpixel += row.num_pixels;
    }
  }

  return totpixel;
}

struct SplitThreadData {
  ThreadQueue *queue;
  ThreadQueue *new_nodes;
  int thread_num, thread_nr;
  std::atomic<bool> *working;

  std::atomic<int> *nodes_num;

  PBVH *pbvh;
  Mesh *mesh;
  Image *image;
  ImageUser *image_user;
};

struct SplitNodePair {
  SplitNodePair *parent;
  PBVHNode node;
  int children_offset = 0;
  int depth = 0;
  int source_index = -1;
  bool is_old = false;

  SplitNodePair(SplitNodePair *node_parent = nullptr) : parent(node_parent)
  {
    memset(static_cast<void *>(&node), 0, sizeof(PBVHNode));
  }
};

static void split_pixel_node(PBVH *pbvh,
                             SplitNodePair *split,
                             Mesh *mesh,
                             Image *image,
                             ImageUser *image_user,
                             SplitThreadData *tdata)
{
  BB cb;
  PBVHNode *node = &split->node;

  cb = node->vb;

  tdata->nodes_num->fetch_add(2);

  if (count_node_pixels(*node) <= pbvh->pixel_leaf_limit || split->depth >= pbvh->depth_limit) {
    return;
  }

  /* Find widest axis and its midpoint */
  const int axis = BB_widest_axis(&cb);
  const float mid = (cb.bmax[axis] + cb.bmin[axis]) * 0.5f;

  node->flag = (PBVHNodeFlags)((int)node->flag & (int)~PBVH_TexLeaf);

  SplitNodePair *split1 = MEM_new<SplitNodePair>("split_pixel_node split1", split);
  SplitNodePair *split2 = MEM_new<SplitNodePair>("split_pixel_node split1", split);

  split1->depth = split->depth + 1;
  split2->depth = split->depth + 1;

  PBVHNode *child1 = &split1->node;
  PBVHNode *child2 = &split2->node;

  child1->flag = PBVH_TexLeaf;
  child2->flag = PBVH_TexLeaf;

  child1->vb = cb;
  child1->vb.bmax[axis] = mid;

  child2->vb = cb;
  child2->vb.bmin[axis] = mid;

  NodeData &data = BKE_pbvh_pixels_node_data_get(split->node);

  NodeData *data1 = MEM_new<NodeData>(__func__);
  NodeData *data2 = MEM_new<NodeData>(__func__);
  child1->pixels.node_data = static_cast<void *>(data1);
  child2->pixels.node_data = static_cast<void *>(data2);

  data1->triangles = data.triangles;
  data2->triangles = data.triangles;

  data1->tiles.resize(data.tiles.size());
  data2->tiles.resize(data.tiles.size());

  for (int i : IndexRange(data.tiles.size())) {
    UDIMTilePixels &tile = data.tiles[i];
    UDIMTilePixels &tile1 = data1->tiles[i];
    UDIMTilePixels &tile2 = data2->tiles[i];

    tile1.tile_number = tile2.tile_number = tile.tile_number;
    tile1.flags.dirty = tile2.flags.dirty = 0;
  }

  ImageUser image_user2 = *image_user;

  for (int i : IndexRange(data.tiles.size())) {
    const UDIMTilePixels &tile = data.tiles[i];

    image_user2.tile = tile.tile_number;

    ImBuf *image_buffer = BKE_image_acquire_ibuf(image, &image_user2, nullptr);
    if (image_buffer == nullptr) {
      continue;
    }

    for (const PackedPixelRow &row : tile.pixel_rows) {
      UDIMTilePixels *tile1 = &data1->tiles[i];
      UDIMTilePixels *tile2 = &data2->tiles[i];

      TrianglePaintInput &tri = data.triangles.paint_input[row.triangle_index];

      float verts[3][3];

      copy_v3_v3(verts[0], mesh->mvert[tri.vert_indices[0]].co);
      copy_v3_v3(verts[1], mesh->mvert[tri.vert_indices[1]].co);
      copy_v3_v3(verts[2], mesh->mvert[tri.vert_indices[2]].co);

      float2 delta = tri.delta_barycentric_coord_u;
      float2 uv1 = row.start_barycentric_coord;
      float2 uv2 = row.start_barycentric_coord + delta * (float)row.num_pixels;

      float co1[3];
      float co2[3];

      interp_barycentric_tri_v3(verts, uv1[0], uv1[1], co1);
      interp_barycentric_tri_v3(verts, uv2[0], uv2[1], co2);

      /* Are we spanning the midpoint? */
      if ((co1[axis] <= mid) != (co2[axis] <= mid)) {
        PackedPixelRow row1 = row;
        float t;

        if (mid < co1[axis]) {
          t = 1.0f - (mid - co2[axis]) / (co1[axis] - co2[axis]);

          SWAP(UDIMTilePixels *, tile1, tile2);
        }
        else {
          t = (mid - co1[axis]) / (co2[axis] - co1[axis]);
        }

        int num_pixels = (int)floorf((float)row.num_pixels * t);

        if (num_pixels) {
          row1.num_pixels = num_pixels;
          tile1->pixel_rows.append(row1);
        }

        if (num_pixels != row.num_pixels) {
          PackedPixelRow row2 = row;

          row2.num_pixels = row.num_pixels - num_pixels;

          row2.start_barycentric_coord = row.start_barycentric_coord +
                                         tri.delta_barycentric_coord_u * (float)num_pixels;
          row2.start_image_coordinate = row.start_image_coordinate;
          row2.start_image_coordinate[0] += num_pixels;

          tile2->pixel_rows.append(row2);
        }
      }
      else if (co1[axis] <= mid && co2[axis] <= mid) {
        tile1->pixel_rows.append(row);
      }
      else {
        tile2->pixel_rows.append(row);
      }
    }

    BKE_image_release_ibuf(image, image_buffer, nullptr);
  }

  pbvh_pixels_free(node);

  BLI_thread_queue_push(tdata->new_nodes, static_cast<void *>(split1));
  BLI_thread_queue_push(tdata->new_nodes, static_cast<void *>(split2));

  BLI_thread_queue_push(tdata->queue, static_cast<void *>(split1));
  BLI_thread_queue_push(tdata->queue, static_cast<void *>(split2));
}

static void split_flush_final_nodes(SplitThreadData *tdata)
{
  PBVH *pbvh = tdata->pbvh;
  Vector<SplitNodePair *> splits;

  while (!BLI_thread_queue_is_empty(tdata->new_nodes)) {
    SplitNodePair *newsplit = static_cast<SplitNodePair *>(BLI_thread_queue_pop(tdata->new_nodes));

    splits.append(newsplit);

    if (newsplit->is_old) {
      continue;
    }

    if (!newsplit->parent->children_offset) {
      newsplit->parent->children_offset = pbvh->totnode;

      pbvh_grow_nodes(pbvh, pbvh->totnode + 2);
      newsplit->source_index = newsplit->parent->children_offset;
    }
    else {
      newsplit->source_index = newsplit->parent->children_offset + 1;
    }
  }

  for (SplitNodePair *split : splits) {
    BLI_assert(split->source_index != -1);

    split->node.children_offset = split->children_offset;
    pbvh->nodes[split->source_index] = split->node;
  }

  for (SplitNodePair *split : splits) {
    MEM_delete<SplitNodePair>(split);
  }
}

extern "C" static void *split_thread_job(void *_tdata)
{
  SplitThreadData *tdata = static_cast<SplitThreadData *>(_tdata);
  int thread_nr = tdata->thread_nr;

  tdata->working[thread_nr].store(false);

  while (1) {
    void *work = BLI_thread_queue_pop_timeout(tdata->queue, 1);

    if (work) {
      SplitNodePair *split = static_cast<SplitNodePair *>(work);

      tdata->working[thread_nr].store(true);
      split_pixel_node(tdata->pbvh, split, tdata->mesh, tdata->image, tdata->image_user, tdata);
      tdata->working[thread_nr].store(false);
      continue;
    }

    bool ok = true;

    for (int i : IndexRange(tdata->thread_num)) {
      if (tdata->working[i].load()) {
        ok = false;
        break;
      }
    }

    if (ok) {
      break;
    }
  }

  return nullptr;
}

static void split_pixel_nodes(PBVH *pbvh, Mesh *mesh, Image *image, ImageUser *image_user)
{
  if (G.debug_value == 891) {
    return;
  }

  if (!pbvh->depth_limit) {
    pbvh->depth_limit = 40; /* TODO: move into a constant */
  }

  if (!pbvh->pixel_leaf_limit) {
    pbvh->pixel_leaf_limit = 256 * 256; /* TODO: move into a constant */
  }

  SplitThreadData tdata;

  tdata.nodes_num = MEM_new<std::atomic<int>>("tdata.nodes_num");
  tdata.nodes_num->store(pbvh->totnode);
  tdata.pbvh = pbvh;
  tdata.mesh = mesh;
  tdata.image = image;
  tdata.image_user = image_user;

  tdata.queue = BLI_thread_queue_init();
  tdata.new_nodes = BLI_thread_queue_init();

  /* Set up initial jobs before initializing threads. */
  for (int i : IndexRange(pbvh->totnode)) {
    if (pbvh->nodes[i].flag & PBVH_TexLeaf) {
      SplitNodePair *split = MEM_new<SplitNodePair>("split_pixel_nodes split");

      split->source_index = i;
      split->is_old = true;
      split->node = pbvh->nodes[i];

      BLI_thread_queue_push(tdata.queue, static_cast<void *>(split));
      BLI_thread_queue_push(tdata.new_nodes, static_cast<void *>(split));
    }
  }

  Vector<SplitThreadData> tdatas;
  int thread_num = tdata.thread_num = G.debug_value != 892 ? BLI_system_thread_count() : 1;

  tdata.working = new std::atomic<bool>[thread_num];

  ListBase threads;
  BLI_threadpool_init(&threads, split_thread_job, thread_num);

  for (int i : IndexRange(thread_num)) {
    tdata.thread_nr = i;
    tdatas.append(tdata);
  }

  for (int i : IndexRange(thread_num)) {
    BLI_threadpool_insert(&threads, &tdatas[i]);
  }

  BLI_threadpool_end(&threads);
  split_flush_final_nodes(&tdata);

  delete[] tdata.working;

  BLI_thread_queue_free(tdata.queue);
  BLI_thread_queue_free(tdata.new_nodes);

  MEM_delete<std::atomic<int>>(tdata.nodes_num);
}

/**
 * During debugging this check could be enabled.
 * It will write to each image pixel that is covered by the PBVH.
 */
constexpr bool USE_WATERTIGHT_CHECK = false;

static void extract_barycentric_pixels(UDIMTilePixels &tile_data,
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
    PackedPixelRow pixel_row;
    pixel_row.triangle_index = triangle_index;
    pixel_row.num_pixels = 0;
    int x;

    for (x = minx; x < maxx; x++) {
      float2 uv(float(x) / image_buffer->x, float(y) / image_buffer->y);
      float3 barycentric_weights;
      barycentric_weights_v2(uvs[0], uvs[1], uvs[2], uv, barycentric_weights);

      const bool is_inside = barycentric_inside_triangle_v2(barycentric_weights);
      if (!start_detected && is_inside) {
        start_detected = true;
        pixel_row.start_image_coordinate = ushort2(x, y);
        pixel_row.start_barycentric_coord = float2(barycentric_weights.x, barycentric_weights.y);
      }
      else if (start_detected && !is_inside) {
        break;
      }
    }

    if (!start_detected) {
      continue;
    }
    pixel_row.num_pixels = x - pixel_row.start_image_coordinate.x;
    tile_data.pixel_rows.append(pixel_row);
  }
}

static void init_triangles(PBVH *pbvh, PBVHNode *node, NodeData *node_data, const MLoop *mloop)
{
  for (int i = 0; i < node->totprim; i++) {
    const MLoopTri *lt = &pbvh->looptri[node->prim_indices[i]];
    node_data->triangles.append(
        int3(mloop[lt->tri[0]].v, mloop[lt->tri[1]].v, mloop[lt->tri[2]].v));
  }
}

struct EncodePixelsUserData {
  Image *image;
  ImageUser *image_user;
  PBVH *pbvh;
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
  PBVH *pbvh = data->pbvh;
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
    UDIMTilePixels tile_data;

    Triangles &triangles = node_data->triangles;
    for (int triangle_index = 0; triangle_index < triangles.size(); triangle_index++) {
      const MLoopTri *lt = &pbvh->looptri[node->prim_indices[triangle_index]];
      float2 uvs[3] = {
          float2(data->ldata_uv[lt->tri[0]].uv) - tile_offset,
          float2(data->ldata_uv[lt->tri[1]].uv) - tile_offset,
          float2(data->ldata_uv[lt->tri[2]].uv) - tile_offset,
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
      triangle.delta_barycentric_coord_u = calc_barycentric_delta_x(image_buffer, uvs, minx, miny);
      extract_barycentric_pixels(
          tile_data, image_buffer, triangle_index, uvs, minx, miny, maxx, maxy);
    }

    BKE_image_release_ibuf(image, image_buffer, nullptr);

    if (tile_data.pixel_rows.is_empty()) {
      continue;
    }

    tile_data.tile_number = image_tile.get_tile_number();
    node_data->tiles.append(tile_data);
  }
}

static bool should_pixels_be_updated(PBVHNode *node)
{
  if ((node->flag & PBVH_Leaf) == 0) {
    return false;
  }
  if (node->children_offset != 0) {
    return false;
  }
  if ((node->flag & PBVH_RebuildPixels) != 0) {
    return true;
  }
  NodeData *node_data = static_cast<NodeData *>(node->pixels.node_data);
  if (node_data != nullptr) {
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
static bool find_nodes_to_update(PBVH *pbvh, Vector<PBVHNode *> &r_nodes_to_update)
{
  int64_t nodes_to_update_len = count_nodes_to_update(pbvh);
  if (nodes_to_update_len == 0) {
    return false;
  }

  r_nodes_to_update.reserve(nodes_to_update_len);

  for (int n = 0; n < pbvh->totnode; n++) {
    PBVHNode *node = &pbvh->nodes[n];
    if (!should_pixels_be_updated(node)) {
      continue;
    }
    r_nodes_to_update.append(node);
    node->flag = static_cast<PBVHNodeFlags>(node->flag | PBVH_RebuildPixels);

    if (node->pixels.node_data == nullptr) {
      NodeData *node_data = MEM_new<NodeData>(__func__);
      node->pixels.node_data = node_data;
    }
    else {
      NodeData *node_data = static_cast<NodeData *>(node->pixels.node_data);
      node_data->clear_data();
    }
  }

  return true;
}

static void apply_watertight_check(PBVH *pbvh, Image *image, ImageUser *image_user)
{
  ImageUser watertight = *image_user;
  LISTBASE_FOREACH (ImageTile *, tile_data, &image->tiles) {
    image::ImageTileWrapper image_tile(tile_data);
    watertight.tile = image_tile.get_tile_number();
    ImBuf *image_buffer = BKE_image_acquire_ibuf(image, &watertight, nullptr);
    if (image_buffer == nullptr) {
      continue;
    }
    for (int n = 0; n < pbvh->totnode; n++) {
      PBVHNode *node = &pbvh->nodes[n];
      if ((node->flag & PBVH_Leaf) == 0) {
        continue;
      }
      NodeData *node_data = static_cast<NodeData *>(node->pixels.node_data);
      UDIMTilePixels *tile_node_data = node_data->find_tile_data(image_tile);
      if (tile_node_data == nullptr) {
        continue;
      }

      for (PackedPixelRow &pixel_row : tile_node_data->pixel_rows) {
        int pixel_offset = pixel_row.start_image_coordinate.y * image_buffer->x +
                           pixel_row.start_image_coordinate.x;
        for (int x = 0; x < pixel_row.num_pixels; x++) {
          if (image_buffer->rect_float) {
            copy_v4_fl(&image_buffer->rect_float[pixel_offset * 4], 1.0);
          }
          if (image_buffer->rect) {
            uint8_t *dest = static_cast<uint8_t *>(
                static_cast<void *>(&image_buffer->rect[pixel_offset]));
            copy_v4_uchar(dest, 255);
          }
          pixel_offset += 1;
        }
      }
    }
    BKE_image_release_ibuf(image, image_buffer, nullptr);
  }
  BKE_image_partial_update_mark_full_update(image);
}

static bool update_pixels(PBVH *pbvh, Mesh *mesh, Image *image, ImageUser *image_user)
{
  Vector<PBVHNode *> nodes_to_update;

  if (!find_nodes_to_update(pbvh, nodes_to_update)) {
    return false;
  }

  MLoopUV *ldata_uv = static_cast<MLoopUV *>(CustomData_get_layer(&mesh->ldata, CD_MLOOPUV));
  if (ldata_uv == nullptr) {
    return false;
  }

  for (PBVHNode *node : nodes_to_update) {
    NodeData *node_data = static_cast<NodeData *>(node->pixels.node_data);
    init_triangles(pbvh, node, node_data, mesh->mloop);
  }

  EncodePixelsUserData user_data;
  user_data.pbvh = pbvh;
  user_data.image = image;
  user_data.image_user = image_user;
  user_data.ldata_uv = ldata_uv;
  user_data.nodes = &nodes_to_update;

  TaskParallelSettings settings;
  BKE_pbvh_parallel_range_settings(&settings, true, nodes_to_update.size());
  BLI_task_parallel_range(0, nodes_to_update.size(), &user_data, do_encode_pixels, &settings);
  if (USE_WATERTIGHT_CHECK) {
    apply_watertight_check(pbvh, image, image_user);
  }

  /* Clear the UpdatePixels flag. */
  for (PBVHNode *node : nodes_to_update) {
    node->flag = static_cast<PBVHNodeFlags>(node->flag & ~PBVH_RebuildPixels);
  }

  /* Add PBVH_TexLeaf flag */
  for (int i : IndexRange(pbvh->totnode)) {
    PBVHNode &node = pbvh->nodes[i];

    if (node.flag & PBVH_Leaf) {
      node.flag = (PBVHNodeFlags)((int)node.flag | (int)PBVH_TexLeaf);
    }
  }

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
      for (const UDIMTilePixels &tile_data : node_data->tiles) {
        compressed_data_len += tile_data.encoded_pixels.size() * sizeof(PackedPixelRow);
        for (const PackedPixelRow &encoded_pixels : tile_data.encoded_pixels) {
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

  return true;
}

NodeData &BKE_pbvh_pixels_node_data_get(PBVHNode &node)
{
  BLI_assert(node.pixels.node_data != nullptr);
  NodeData *node_data = static_cast<NodeData *>(node.pixels.node_data);
  return *node_data;
}

void BKE_pbvh_pixels_mark_image_dirty(PBVHNode &node, Image &image, ImageUser &image_user)
{
  BLI_assert(node.pixels.node_data != nullptr);
  NodeData *node_data = static_cast<NodeData *>(node.pixels.node_data);
  if (node_data->flags.dirty) {
    ImageUser local_image_user = image_user;
    LISTBASE_FOREACH (ImageTile *, tile, &image.tiles) {
      image::ImageTileWrapper image_tile(tile);
      local_image_user.tile = image_tile.get_tile_number();
      ImBuf *image_buffer = BKE_image_acquire_ibuf(&image, &local_image_user, nullptr);
      if (image_buffer == nullptr) {
        continue;
      }

      node_data->mark_region(image, image_tile, *image_buffer);
      BKE_image_release_ibuf(&image, image_buffer, nullptr);
    }
    node_data->flags.dirty = false;
  }
}
}  // namespace blender::bke::pbvh::pixels

extern "C" {
using namespace blender::bke::pbvh::pixels;

void BKE_pbvh_build_pixels(PBVH *pbvh, Mesh *mesh, Image *image, ImageUser *image_user)
{
  if (update_pixels(pbvh, mesh, image, image_user)) {
    double time_ms = PIL_check_seconds_timer();

    split_pixel_nodes(pbvh, mesh, image, image_user);

    time_ms = PIL_check_seconds_timer() - time_ms;

    printf("Nodes split time: %.2fms\n", time_ms * 1000.0);
  }
}

void pbvh_pixels_free(PBVHNode *node)
{
  NodeData *node_data = static_cast<NodeData *>(node->pixels.node_data);
  MEM_delete(node_data);
  node->pixels.node_data = nullptr;
}
}
