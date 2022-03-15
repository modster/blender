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

#include "BLI_math_color_blend.h"
#include "BLI_task.h"
#include "BLI_vector.hh"

#include "IMB_rasterizer.hh"

#include "WM_types.h"

#include "bmesh.h"

#include "ED_uvedit.h"

#include "sculpt_intern.h"
#include "sculpt_texture_paint_intern.hh"

namespace blender::ed::sculpt_paint::texture_paint {
namespace painting {

static Pixel get_start_pixel(const PixelsPackage &encoded_pixels,
                             const Triangle &triangle,
                             const MVert *mvert,
                             const MLoopUV *ldata_uv)
{
  Pixel result;
  const float3 weights = encoded_pixels.start_edge_coord;
  interp_v3_v3v3v3(result.pos,
                   mvert[triangle.vert_indices[0]].co,
                   mvert[triangle.vert_indices[1]].co,
                   mvert[triangle.vert_indices[2]].co,
                   weights);
  interp_v3_v3v3v3(result.uv,
                   ldata_uv[triangle.loop_indices[0]].uv,
                   ldata_uv[triangle.loop_indices[1]].uv,
                   ldata_uv[triangle.loop_indices[2]].uv,
                   weights);

  return result;
}

static Pixel get_delta_pixel(const PixelsPackage &encoded_pixels,
                             const Triangle &triangle,
                             const Pixel &start_pixel,
                             const MVert *mvert,
                             const MLoopUV *ldata_uv

)
{
  Pixel result;
  const float3 weights = encoded_pixels.start_edge_coord + triangle.add_edge_coord_x;
  interp_v3_v3v3v3(result.pos,
                   mvert[triangle.vert_indices[0]].co,
                   mvert[triangle.vert_indices[1]].co,
                   mvert[triangle.vert_indices[2]].co,
                   weights);
  interp_v3_v3v3v3(result.uv,
                   ldata_uv[triangle.loop_indices[0]].uv,
                   ldata_uv[triangle.loop_indices[1]].uv,
                   ldata_uv[triangle.loop_indices[2]].uv,
                   weights);
  result.pos -= start_pixel.pos;
  result.uv -= start_pixel.uv;
  return result;
}

static void add(Pixel &dst, const Pixel &lh)
{
  dst.pos += lh.pos;
  dst.uv += lh.uv;
}

static void do_vertex_brush_test(void *__restrict userdata,
                                 const int n,
                                 const TaskParallelTLS *__restrict UNUSED(tls))
{
  TexturePaintingUserData *data = static_cast<TexturePaintingUserData *>(userdata);
  const Object *ob = data->ob;
  SculptSession *ss = ob->sculpt;
  const Brush *brush = data->brush;
  SculptBrushTest test;
  SculptBrushTestFn sculpt_brush_test_sq_fn = SCULPT_brush_test_init_with_falloff_shape(
      ss, &test, brush->falloff_shape);
  PBVHNode *node = data->nodes[n];

  PBVHVertexIter vd;
  BKE_pbvh_vertex_iter_begin (ss->pbvh, node, vd, PBVH_ITER_UNIQUE) {
    data->vertex_brush_tests[vd.index] = sculpt_brush_test_sq_fn(&test, vd.co);
  }
  BKE_pbvh_vertex_iter_end;
}

static void do_task_cb_ex(void *__restrict userdata,
                          const int n,
                          const TaskParallelTLS *__restrict tls)
{
  TexturePaintingUserData *data = static_cast<TexturePaintingUserData *>(userdata);
  Object *ob = data->ob;
  SculptSession *ss = ob->sculpt;
  ImBuf *image_buffer = ss->mode.texture_paint.drawing_target;
  const Brush *brush = data->brush;
  PBVHNode *node = data->nodes[n];
  NodeData *node_data = static_cast<NodeData *>(BKE_pbvh_node_texture_paint_data_get(node));
  BLI_assert(node_data != nullptr);

  const int thread_id = BLI_task_parallel_thread_id(tls);

  SculptBrushTest test;
  SculptBrushTestFn sculpt_brush_test_sq_fn = SCULPT_brush_test_init_with_falloff_shape(
      ss, &test, brush->falloff_shape);

  float3 brush_srgb(brush->rgb[0], brush->rgb[1], brush->rgb[2]);
  float4 brush_linear;
  srgb_to_linearrgb_v3_v3(brush_linear, brush_srgb);
  brush_linear[3] = 1.0f;
  MVert *mvert = SCULPT_mesh_deformed_mverts_get(ss);
  Mesh *mesh = static_cast<Mesh *>(ob->data);
  MLoopUV *ldata_uv = static_cast<MLoopUV *>(CustomData_get_layer(&mesh->ldata, CD_MLOOPUV));

  /* Propagate vertex brush test to triangle. This should be extended with brush overlapping edges
   * and faces only. */
  std::vector<bool> triangle_brush_test_results(node_data->triangles.size());
  int triangle_index = 0;
  for (Triangle &triangle : node_data->triangles) {
    for (int i = 0; i < 3; i++) {
      triangle_brush_test_results[triangle_index] =
          triangle_brush_test_results[triangle_index] ||
          data->vertex_brush_tests[triangle.vert_indices[i]];
    }
    triangle_index += 1;
  }

  const float brush_strength = ss->cache->bstrength;
  int packages_clipped = 0;

  for (PixelsPackage &encoded_pixels : node_data->encoded_pixels) {
    if (!triangle_brush_test_results[encoded_pixels.triangle_index]) {
      packages_clipped += 1;
      continue;
    }
    Triangle &triangle = node_data->triangles[encoded_pixels.triangle_index];
    int pixel_offset = encoded_pixels.start_image_coordinate.y * image_buffer->x +
                       encoded_pixels.start_image_coordinate.x;
    float3 edge_coord = encoded_pixels.start_edge_coord;
    Pixel pixel = get_start_pixel(encoded_pixels, triangle, mvert, ldata_uv);
    const Pixel add_pixel = get_delta_pixel(encoded_pixels, triangle, pixel, mvert, ldata_uv);
    bool pixels_painted = false;
    for (int x = 0; x < encoded_pixels.num_pixels; x++) {
      if (!sculpt_brush_test_sq_fn(&test, pixel.pos)) {
        add(pixel, add_pixel);
        pixel_offset += 1;
        continue;
      }

      float *color = &image_buffer->rect_float[pixel_offset * 4];
      const float3 normal(0.0f, 0.0f, 0.0f);
      const float3 face_normal(0.0f, 0.0f, 0.0f);
      const float mask = 0.0f;
      const float falloff_strength = SCULPT_brush_strength_factor(
          ss, brush, pixel.pos, sqrtf(test.dist), normal, face_normal, mask, 0, thread_id);

      blend_color_interpolate_float(color, color, brush_linear, falloff_strength * brush_strength);
      pixels_painted = true;

      edge_coord += triangle.add_edge_coord_x;
      add(pixel, add_pixel);
      pixel_offset++;
    }

    if (pixels_painted) {
      BLI_rcti_do_minmax_v(&node_data->dirty_region, encoded_pixels.start_image_coordinate);
      BLI_rcti_do_minmax_v(
          &node_data->dirty_region,
          int2(encoded_pixels.start_image_coordinate.x + encoded_pixels.num_pixels + 1,
               encoded_pixels.start_image_coordinate.y));
      node_data->flags.dirty = true;
    }
  }
  printf("%d of %ld pixel packages clipped\n", packages_clipped, node_data->encoded_pixels.size());
}
}  // namespace painting

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

  Mesh *mesh = (Mesh *)ob->data;

  TexturePaintingUserData data = {nullptr};
  data.ob = ob;
  data.brush = brush;
  data.nodes = nodes;
  data.vertex_brush_tests = std::vector<bool>(mesh->totvert);

  TaskParallelSettings settings;
  BKE_pbvh_parallel_range_settings(&settings, true, totnode);

  TIMEIT_START(texture_painting);
  BLI_task_parallel_range(0, totnode, &data, painting::do_vertex_brush_test, &settings);
  BLI_task_parallel_range(0, totnode, &data, painting::do_task_cb_ex, &settings);
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
  SCULPT_extract_pixels(ob, nodes, totnode);

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
      data->mark_region(*image_data.image, *image_data.image_buffer);
      data->flags.dirty = false;
    }
  }

  MEM_freeN(nodes);
}
}
}  // namespace blender::ed::sculpt_paint::texture_paint
