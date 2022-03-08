/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

/** \file
 * \ingroup edsculpt
 */

#include "DNA_material_types.h"
#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"
#include "DNA_scene_types.h"
#include "DNA_windowmanager_types.h"

#include "BKE_brush.h"
#include "BKE_context.h"
#include "BKE_customdata.h"
#include "BKE_image.h"
#include "BKE_material.h"
#include "BKE_mesh.h"
#include "BKE_mesh_mapping.h"
#include "BKE_pbvh.h"

#include "PIL_time_utildefines.h"

#include "BLI_task.h"
#include "BLI_vector.hh"

#include "IMB_rasterizer.hh"

#include "WM_types.h"

#include "bmesh.h"

#include "ED_uvedit.h"

#include "sculpt_intern.h"

namespace blender::ed::sculpt_paint::texture_paint {

struct PixelData {
  struct {
    bool dirty : 1;
  } flags;
  int2 pixel_pos;
  float3 local_pos;
  float4 content;
};

struct NodeData {
  struct {
    bool dirty : 1;
  } flags;

  Vector<PixelData> pixels;
  rcti dirty_region;
  rctf uv_region;

  NodeData()
  {
    flags.dirty = false;
    BLI_rcti_init_minmax(&dirty_region);
  }

  void init_pixels_rasterization(Object *ob, PBVHNode *node, ImBuf *image_buffer);

  void flush(ImBuf &image_buffer)
  {
    flags.dirty = false;
    for (PixelData &pixel : pixels) {
      if (pixel.flags.dirty) {
        const int pixel_offset = (pixel.pixel_pos[1] * image_buffer.x + pixel.pixel_pos[0]) * 4;
        copy_v4_v4(&image_buffer.rect_float[pixel_offset], pixel.content);
        pixel.flags.dirty = false;
      }
    }
  }

  void mark_region(Image &image, ImBuf &image_buffer)
  {
    printf("%s", __func__);
    print_rcti_id(&dirty_region);
    BKE_image_partial_update_mark_region(
        &image, static_cast<ImageTile *>(image.tiles.first), &image_buffer, &dirty_region);
    BLI_rcti_init_minmax(&dirty_region);
  }

  static void free_func(void *instance)
  {
    NodeData *node_data = static_cast<NodeData *>(instance);
    MEM_delete(node_data);
  }
};

namespace shaders {

using namespace imbuf::rasterizer;

struct VertexInput {
  float3 pos;
  float2 uv;

  VertexInput(float3 pos, float2 uv) : pos(pos), uv(uv)
  {
  }
};

class VertexShader : public AbstractVertexShader<VertexInput, float3> {
 public:
  float2 image_size;
  void vertex(const VertexInputType &input, VertexOutputType *r_output) override
  {
    r_output->coord = input.uv * image_size;
    r_output->data = input.pos;
  }
};

struct FragmentOutput {
  float3 local_pos;
};

class FragmentShader : public AbstractFragmentShader<float3, FragmentOutput> {
 public:
  ImBuf *image_buffer;

 public:
  void fragment(const FragmentInputType &input, FragmentOutputType *r_output) override
  {
    r_output->local_pos = input;
  }
};

struct NodeDataPair {
  ImBuf *image_buffer;
  NodeData *node_data;

  struct {
    /* Rasterizer doesn't support glCoord yet, so for now we just store them in a runtime section.
     */
    int2 last_known_pixel_pos;
  } runtime;
};

class AddPixel : public AbstractBlendMode<FragmentOutput, NodeDataPair> {
 public:
  void blend(NodeDataPair *dest, const FragmentOutput &source) const override
  {
    PixelData new_pixel;
    new_pixel.local_pos = source.local_pos;
    new_pixel.pixel_pos = dest->runtime.last_known_pixel_pos;
    const int pixel_offset = new_pixel.pixel_pos[1] * dest->image_buffer->x +
                             new_pixel.pixel_pos[0];
    new_pixel.content = float4(dest->image_buffer->rect_float[pixel_offset * 4]);
    new_pixel.flags.dirty = false;

    dest->node_data->pixels.append(new_pixel);
    dest->runtime.last_known_pixel_pos[0] += 1;
  }
};

class NodeDataDrawingTarget : public AbstractDrawingTarget<NodeDataPair, NodeDataPair> {
 private:
  NodeDataPair *active_ = nullptr;

 public:
  uint64_t get_width() const
  {
    return active_->image_buffer->x;
  }
  uint64_t get_height() const
  {
    return active_->image_buffer->y;
  };
  NodeDataPair *get_pixel_ptr(uint64_t x, uint64_t y)
  {
    active_->runtime.last_known_pixel_pos = int2(x, y);
    return active_;
  };
  int64_t get_pixel_stride() const
  {
    return 0;
  };
  bool has_active_target() const
  {
    return active_ != nullptr;
  }
  void activate(NodeDataPair *instance)
  {
    active_ = instance;
  };
  void deactivate()
  {
    active_ = nullptr;
  }
};

using RasterizerType = Rasterizer<VertexShader, FragmentShader, AddPixel, NodeDataDrawingTarget>;

}  // namespace shaders

void NodeData::init_pixels_rasterization(Object *ob, PBVHNode *node, ImBuf *image_buffer)
{
  Mesh *mesh = static_cast<Mesh *>(ob->data);
  MLoopUV *ldata_uv = static_cast<MLoopUV *>(CustomData_get_layer(&mesh->ldata, CD_MLOOPUV));
  if (ldata_uv == nullptr) {
    return;
  }

  shaders::RasterizerType rasterizer;
  shaders::NodeDataPair node_data_pair;
  rasterizer.vertex_shader().image_size = float2(image_buffer->x, image_buffer->y);
  rasterizer.fragment_shader().image_buffer = image_buffer;
  node_data_pair.node_data = this;
  node_data_pair.image_buffer = image_buffer;
  rasterizer.activate_drawing_target(&node_data_pair);

  SculptSession *ss = ob->sculpt;
  MVert *mvert = SCULPT_mesh_deformed_mverts_get(ss);

  PBVHVertexIter vd;
  BKE_pbvh_vertex_iter_begin (ss->pbvh, node, vd, PBVH_ITER_UNIQUE) {
    MeshElemMap *vert_map = &ss->pmap[vd.index];
    for (int j = 0; j < ss->pmap[vd.index].count; j++) {
      const MPoly *p = &ss->mpoly[vert_map->indices[j]];
      if (p->totloop < 3) {
        continue;
      }

      const MLoop *loopstart = &ss->mloop[p->loopstart];
      for (int triangle = 0; triangle < p->totloop - 2; triangle++) {
        const int v1_index = loopstart[0].v;
        const int v2_index = loopstart[triangle + 1].v;
        const int v3_index = loopstart[triangle + 2].v;
        const int v1_loop_index = p->loopstart;
        const int v2_loop_index = p->loopstart + triangle + 1;
        const int v3_loop_index = p->loopstart + triangle + 2;

        shaders::VertexInput v1(mvert[v1_index].co, ldata_uv[v1_loop_index].uv);
        shaders::VertexInput v2(mvert[v2_index].co, ldata_uv[v2_loop_index].uv);
        shaders::VertexInput v3(mvert[v3_index].co, ldata_uv[v3_loop_index].uv);
        rasterizer.draw_triangle(v1, v2, v3);
      }
    }
  }
  BKE_pbvh_vertex_iter_end;
  rasterizer.deactivate_drawing_target();
}

struct TexturePaintingUserData {
  Object *ob;
  Brush *brush;
  PBVHNode **nodes;
};

static void do_task_cb_ex(void *__restrict userdata,
                          const int n,
                          const TaskParallelTLS *__restrict UNUSED(tls))
{
  TexturePaintingUserData *data = static_cast<TexturePaintingUserData *>(userdata);
  Object *ob = data->ob;
  SculptSession *ss = ob->sculpt;
  const Brush *brush = data->brush;
  PBVHNode *node = data->nodes[n];
  NodeData *node_data = static_cast<NodeData *>(BKE_pbvh_node_texture_paint_data_get(node));
  BLI_assert(node_data != nullptr);

  SculptBrushTest test;
  SculptBrushTestFn sculpt_brush_test_sq_fn = SCULPT_brush_test_init_with_falloff_shape(
      ss, &test, brush->falloff_shape);

  for (PixelData &pixel : node_data->pixels) {
    if (!sculpt_brush_test_sq_fn(&test, pixel.local_pos)) {
      continue;
    }
    const float falloff_strength = BKE_brush_curve_strength(brush, sqrtf(test.dist), test.radius);
    interp_v3_v3v3(pixel.content, pixel.content, brush->rgb, falloff_strength);
    pixel.content[3] = 1.0f;
    pixel.flags.dirty = true;
    BLI_rcti_do_minmax_v(&node_data->dirty_region, pixel.pixel_pos);
    node_data->flags.dirty = true;
  }
}

static void init_rasterization_task_cb_ex(void *__restrict userdata,
                                          const int n,
                                          const TaskParallelTLS *__restrict UNUSED(tls))
{
  TexturePaintingUserData *data = static_cast<TexturePaintingUserData *>(userdata);
  Object *ob = data->ob;
  SculptSession *ss = ob->sculpt;
  PBVHNode *node = data->nodes[n];

  NodeData *node_data = static_cast<NodeData *>(BKE_pbvh_node_texture_paint_data_get(node));
  // TODO: reinit when texturing on different image?
  if (node_data != nullptr) {
    return;
  }

  TIMEIT_START(init_texture_paint_for_node);
  node_data = MEM_new<NodeData>(__func__);
  node_data->init_pixels_rasterization(ob, node, ss->mode.texture_paint.drawing_target);
  BKE_pbvh_node_texture_paint_data_set(node, node_data, NodeData::free_func);
  TIMEIT_END(init_texture_paint_for_node);
}

static void init_using_rasterization(Object *ob, int totnode, PBVHNode **nodes)
{
  TIMEIT_START(init_using_rasterization);
  TexturePaintingUserData data = {nullptr};
  data.ob = ob;
  data.nodes = nodes;

  TaskParallelSettings settings;
  BKE_pbvh_parallel_range_settings(&settings, true, totnode);

  BLI_task_parallel_range(0, totnode, &data, init_rasterization_task_cb_ex, &settings);
  TIMEIT_END(init_using_rasterization);
}

struct BucketEntry {
  PBVHNode *node;
  const MPoly *poly;
};
struct Bucket {
  static const int Size = 16;
  Vector<BucketEntry> entries;
  rctf bounds;
};

static bool init_using_intersection(SculptSession *ss,
                                    Bucket &bucket,
                                    ImBuf *image_buffer,
                                    MVert *mvert,
                                    MLoopUV *ldata_uv,
                                    float2 uv,
                                    int2 xy)
{
  const int pixel_offset = xy[1] * image_buffer->x + xy[0];
  for (BucketEntry &entry : bucket.entries) {
    const MPoly *p = entry.poly;

    const MLoop *loopstart = &ss->mloop[p->loopstart];
    for (int triangle = 0; triangle < p->totloop - 2; triangle++) {
      const int v1_loop_index = p->loopstart;
      const int v2_loop_index = p->loopstart + triangle + 1;
      const int v3_loop_index = p->loopstart + triangle + 2;
      const float2 v1_uv = ldata_uv[v1_loop_index].uv;
      const float2 v2_uv = ldata_uv[v2_loop_index].uv;
      const float2 v3_uv = ldata_uv[v3_loop_index].uv;
      float3 weights;
      barycentric_weights_v2(v1_uv, v2_uv, v3_uv, uv, weights);
      if (weights[0] < 0.0 || weights[0] > 1.0 || weights[1] < 0.0 || weights[1] > 1.0 ||
          weights[2] < 0.0 || weights[2] > 1.0) {
        continue;
      }

      const int v1_index = loopstart[0].v;
      const int v2_index = loopstart[triangle + 1].v;
      const int v3_index = loopstart[triangle + 2].v;
      const float3 v1_pos = mvert[v1_index].co;
      const float3 v2_pos = mvert[v2_index].co;
      const float3 v3_pos = mvert[v3_index].co;
      float3 local_pos;
      interp_v3_v3v3v3(local_pos, v1_pos, v2_pos, v3_pos, weights);

      PixelData new_pixel;
      new_pixel.local_pos = local_pos;
      new_pixel.pixel_pos = xy;
      copy_v4_fl(&image_buffer->rect_float[pixel_offset * 4], 1.0);
      new_pixel.content = float4(image_buffer->rect_float[pixel_offset * 4]);
      new_pixel.flags.dirty = false;

      PBVHNode *node = entry.node;
      NodeData *node_data = static_cast<NodeData *>(BKE_pbvh_node_texture_paint_data_get(node));
      node_data->pixels.append(new_pixel);
      return true;
    }
  }
  return false;
}

static bool init_using_intersection(SculptSession *ss,
                                    PBVHNode *node,
                                    NodeData *node_data,
                                    ImBuf *image_buffer,
                                    MVert *mvert,
                                    MLoopUV *ldata_uv,
                                    float2 uv,
                                    int2 xy)
{
  const int pixel_offset = xy[1] * image_buffer->x + xy[0];
  PBVHVertexIter vd;
  BKE_pbvh_vertex_iter_begin (ss->pbvh, node, vd, PBVH_ITER_UNIQUE) {
    MeshElemMap *vert_map = &ss->pmap[vd.index];
    for (int j = 0; j < ss->pmap[vd.index].count; j++) {
      const MPoly *p = &ss->mpoly[vert_map->indices[j]];
      if (p->totloop < 3) {
        continue;
      }

      const MLoop *loopstart = &ss->mloop[p->loopstart];
      for (int triangle = 0; triangle < p->totloop - 2; triangle++) {
        const int v1_loop_index = p->loopstart;
        const int v2_loop_index = p->loopstart + triangle + 1;
        const int v3_loop_index = p->loopstart + triangle + 2;
        const float2 v1_uv = ldata_uv[v1_loop_index].uv;
        const float2 v2_uv = ldata_uv[v2_loop_index].uv;
        const float2 v3_uv = ldata_uv[v3_loop_index].uv;
        float3 weights;
        barycentric_weights_v2(v1_uv, v2_uv, v3_uv, uv, weights);
        if (weights[0] < 0.0 || weights[0] > 1.0 || weights[1] < 0.0 || weights[1] > 1.0 ||
            weights[2] < 0.0 || weights[2] > 1.0) {
          continue;
        }

        const int v1_index = loopstart[0].v;
        const int v2_index = loopstart[triangle + 1].v;
        const int v3_index = loopstart[triangle + 2].v;
        const float3 v1_pos = mvert[v1_index].co;
        const float3 v2_pos = mvert[v2_index].co;
        const float3 v3_pos = mvert[v3_index].co;
        float3 local_pos;
        interp_v2_v2v2v2(local_pos, v1_pos, v2_pos, v3_pos, weights);

        PixelData new_pixel;
        new_pixel.local_pos = local_pos;
        new_pixel.pixel_pos = xy;
        new_pixel.content = float4(image_buffer->rect_float[pixel_offset * 4]);
        copy_v4_fl(&image_buffer->rect_float[pixel_offset * 4], 1.0);
        new_pixel.flags.dirty = false;
        node_data->pixels.append(new_pixel);
        return true;
      }
    }
  }
  BKE_pbvh_vertex_iter_end;

  return false;
}

static void init_using_intersection(Object *ob, int totnode, PBVHNode **nodes)
{
  TIMEIT_START(init_using_intersection);

  Vector<PBVHNode *> nodes_to_initialize;
  for (int n = 0; n < totnode; n++) {
    PBVHNode *node = nodes[n];
    NodeData *node_data = static_cast<NodeData *>(BKE_pbvh_node_texture_paint_data_get(node));
    if (node_data != nullptr) {
      continue;
    }
    node_data = MEM_new<NodeData>(__func__);
    BKE_pbvh_node_texture_paint_data_set(node, node_data, NodeData::free_func);
    BLI_rctf_init_minmax(&node_data->uv_region);
    nodes_to_initialize.append(node);
  }
  if (nodes_to_initialize.size() == 0) {
    return;
  }

  Mesh *mesh = static_cast<Mesh *>(ob->data);
  MLoopUV *ldata_uv = static_cast<MLoopUV *>(CustomData_get_layer(&mesh->ldata, CD_MLOOPUV));
  if (ldata_uv == nullptr) {
    return;
  }

  SculptSession *ss = ob->sculpt;
  MVert *mvert = SCULPT_mesh_deformed_mverts_get(ss);
  ImBuf *image_buffer = ss->mode.texture_paint.drawing_target;
  int pixels_added = 0;
  Bucket bucket;
  for (int y_bucket = 0; y_bucket < image_buffer->y; y_bucket += Bucket::Size) {
    for (int x_bucket = 0; x_bucket < image_buffer->x; x_bucket += Bucket::Size) {
      bucket.entries.clear();
      BLI_rctf_init(&bucket.bounds,
                    float(x_bucket) / image_buffer->x,
                    float(x_bucket + Bucket::Size) / image_buffer->x,
                    float(y_bucket) / image_buffer->y,
                    float(y_bucket + Bucket::Size) / image_buffer->y);
      print_rctf_id(&bucket.bounds);
      printf("%d pixels already added.\n", pixels_added);

      for (int n = 0; n < nodes_to_initialize.size(); n++) {
        PBVHNode *node = nodes_to_initialize[n];
        PBVHVertexIter vd;
        NodeData *node_data = static_cast<NodeData *>(BKE_pbvh_node_texture_paint_data_get(node));
        if (BLI_rctf_is_valid(&node_data->uv_region)) {
          if (!BLI_rctf_isect(&bucket.bounds, &node_data->uv_region, nullptr)) {
            continue;
          }
        }

        BKE_pbvh_vertex_iter_begin (ss->pbvh, node, vd, PBVH_ITER_UNIQUE) {
          MeshElemMap *vert_map = &ss->pmap[vd.index];
          for (int j = 0; j < ss->pmap[vd.index].count; j++) {
            const MPoly *p = &ss->mpoly[vert_map->indices[j]];
            if (p->totloop < 3) {
              continue;
            }

            rctf poly_bound;
            BLI_rctf_init_minmax(&poly_bound);
            for (int triangle = 0; triangle < p->totloop - 2; triangle++) {
              const int v1_loop_index = p->loopstart;
              const int v2_loop_index = p->loopstart + triangle + 1;
              const int v3_loop_index = p->loopstart + triangle + 2;
              const float2 v1_uv = ldata_uv[v1_loop_index].uv;
              const float2 v2_uv = ldata_uv[v2_loop_index].uv;
              const float2 v3_uv = ldata_uv[v3_loop_index].uv;
              BLI_rctf_do_minmax_v(&poly_bound, v1_uv);
              BLI_rctf_do_minmax_v(&poly_bound, v2_uv);
              BLI_rctf_do_minmax_v(&poly_bound, v3_uv);
              BLI_rctf_do_minmax_v(&node_data->uv_region, v1_uv);
              BLI_rctf_do_minmax_v(&node_data->uv_region, v2_uv);
              BLI_rctf_do_minmax_v(&node_data->uv_region, v3_uv);
            }
            if (BLI_rctf_isect(&bucket.bounds, &poly_bound, nullptr)) {
              BucketEntry entry;
              entry.node = node;
              entry.poly = p;
              bucket.entries.append(entry);
            }
          }
        }
        BKE_pbvh_vertex_iter_end;
      }
      printf("Loaded %ld entries in bucket\n", bucket.entries.size());
      if (bucket.entries.size() == 0) {
        continue;
      }

      for (int y = y_bucket; y < image_buffer->y && y < y_bucket + Bucket::Size; y++) {
        for (int x = x_bucket; x < image_buffer->x && x < x_bucket + Bucket::Size; x++) {
          float2 uv(float(x) / image_buffer->x, float(y) / image_buffer->y);
#if 0
          for (int n = 0; n < nodes_to_initialize.size(); n++) {
            PBVHNode *node = nodes_to_initialize[n];
            NodeData *node_data = static_cast<NodeData *>(
                BKE_pbvh_node_texture_paint_data_get(node));
            if (init_using_intersection(
                    ss, node, node_data, image_buffer, mvert, ldata_uv, uv, int2(x, y))) {
              pixels_added++;
              break;
            }
          }
#else
          if (init_using_intersection(ss, bucket, image_buffer, mvert, ldata_uv, uv, int2(x, y))) {
            pixels_added++;
          }
#endif
        }
      }
    }
  }
  TIMEIT_END(init_using_intersection);
}

struct ImageData {
  void *lock = nullptr;
  Image *image = nullptr;
  ImageUser *image_user = nullptr;
  ImBuf *image_buffer = nullptr;

  ~ImageData()
  {
    BKE_image_release_ibuf(image, image_buffer, lock);
  }

  static bool init_active_image(Object *ob, ImageData *r_image_data)
  {
    ED_object_get_active_image(
        ob, 1, &r_image_data->image, &r_image_data->image_user, nullptr, nullptr);
    if (r_image_data->image == nullptr) {
      return false;
    }
    r_image_data->image_buffer = BKE_image_acquire_ibuf(
        r_image_data->image, r_image_data->image_user, &r_image_data->lock);
    if (r_image_data->image_buffer == nullptr) {
      return false;
    }
    return true;
  }
};

extern "C" {
void SCULPT_do_texture_paint_brush(Sculpt *sd, Object *ob, PBVHNode **nodes, int totnode)
{
  SculptSession *ss = ob->sculpt;
  Brush *brush = BKE_paint_brush(&sd->paint);

  ImageData image_data;
  if (!ImageData::init_active_image(ob, &image_data)) {
    return;
  }

  ss->mode.texture_paint.drawing_target = image_data.image_buffer;

  TexturePaintingUserData data = {nullptr};
  data.ob = ob;
  data.brush = brush;
  data.nodes = nodes;

  TaskParallelSettings settings;
  BKE_pbvh_parallel_range_settings(&settings, true, totnode);

  TIMEIT_START(texture_painting);
  BLI_task_parallel_range(0, totnode, &data, do_task_cb_ex, &settings);
  TIMEIT_END(texture_painting);

  ss->mode.texture_paint.drawing_target = nullptr;
}

void SCULPT_init_texture_paint(Object *ob)
{
  SculptSession *ss = ob->sculpt;
  ImageData image_data;
  if (!ImageData::init_active_image(ob, &image_data)) {
    return;
  }
  ss->mode.texture_paint.drawing_target = image_data.image_buffer;
  PBVHNode **nodes;
  int totnode;
  BKE_pbvh_search_gather(ss->pbvh, NULL, NULL, &nodes, &totnode);
  const bool do_rasterization = false;
  if (do_rasterization) {
    init_using_rasterization(ob, totnode, nodes);
  }
  else {
    init_using_intersection(ob, totnode, nodes);
  }

  MEM_freeN(nodes);

  ss->mode.texture_paint.drawing_target = nullptr;
}

void SCULPT_flush_texture_paint(Object *ob)
{
  ImageData image_data;
  if (!ImageData::init_active_image(ob, &image_data)) {
    return;
  }

  SculptSession *ss = ob->sculpt;
  PBVHNode **nodes;
  int totnode;
  BKE_pbvh_search_gather(ss->pbvh, NULL, NULL, &nodes, &totnode);
  for (int n = 0; n < totnode; n++) {
    PBVHNode *node = nodes[n];
    NodeData *data = static_cast<NodeData *>(BKE_pbvh_node_texture_paint_data_get(node));
    if (data == nullptr) {
      continue;
    }

    if (data->flags.dirty) {
      data->flush(*image_data.image_buffer);
      data->mark_region(*image_data.image, *image_data.image_buffer);
    }
  }

  MEM_freeN(nodes);
}
}
}  // namespace blender::ed::sculpt_paint::texture_paint
