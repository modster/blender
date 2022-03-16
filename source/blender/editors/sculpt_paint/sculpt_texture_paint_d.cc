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
#include "BLI_math_vec_types.hh"
#include "BLI_string_ref.hh"
#include "BLI_task.h"
#include "BLI_vector.hh"

#include "IMB_colormanagement.h"
#include "IMB_imbuf_types.h"

#include "WM_types.h"

#include "bmesh.h"

#include "ED_uvedit.h"

#include "sculpt_intern.h"
#include "sculpt_texture_paint_intern.hh"

namespace blender::ed::sculpt_paint::texture_paint {
namespace painting {

/** Reading and writing to image buffer with 4 float channels. */
class ImageBufferFloat4 {
 private:
  int pixel_offset;

 public:
  void set_image_position(ImBuf *image_buffer, int2 image_pixel_position)
  {
    pixel_offset = image_pixel_position.y * image_buffer->x + image_pixel_position.x;
  }

  void goto_next_pixel()
  {
    pixel_offset += 1;
  }

  float4 read_pixel(ImBuf *image_buffer) const
  {
    return &image_buffer->rect_float[pixel_offset * 4];
  }

  void store_pixel(ImBuf *image_buffer, const float4 pixel_data) const
  {
    copy_v4_v4(&image_buffer->rect_float[pixel_offset * 4], pixel_data);
  }

  const char *get_colorspace_name(ImBuf *image_buffer)
  {
    return IMB_colormanagement_get_float_colorspace(image_buffer);
  }
};

/** Reading and writing to image buffer with 4 byte channels. */
class ImageBufferByte4 {
 private:
  int pixel_offset;

 public:
  void set_image_position(ImBuf *image_buffer, int2 image_pixel_position)
  {
    pixel_offset = image_pixel_position.y * image_buffer->x + image_pixel_position.x;
  }

  void goto_next_pixel()
  {
    pixel_offset += 1;
  }

  float4 read_pixel(ImBuf *image_buffer) const
  {
    float4 result;
    rgba_uchar_to_float(result,
                        static_cast<const uchar *>(
                            static_cast<const void *>(&(image_buffer->rect[pixel_offset]))));
    return result;
  }

  void store_pixel(ImBuf *image_buffer, const float4 pixel_data) const
  {
    rgba_float_to_uchar(
        static_cast<uchar *>(static_cast<void *>(&image_buffer->rect[pixel_offset])), pixel_data);
  }

  const char *get_colorspace_name(ImBuf *image_buffer)
  {
    return IMB_colormanagement_get_rect_colorspace(image_buffer);
  }
};

template<typename ImagePixelAccessor> class PaintingKernel {
  ImagePixelAccessor image_accessor;

  SculptSession *ss;
  const Brush *brush;
  const int thread_id;
  const MVert *mvert;

  float4 brush_color;
  float brush_strength;

  SculptBrushTestFn brush_test_fn;
  SculptBrushTest test;
  /* Pointer to the last used image buffer to detect when buffers are switched. */
  void *last_used_image_buffer_ptr = nullptr;
  const char *last_used_color_space = nullptr;

 public:
  explicit PaintingKernel(SculptSession *ss,
                          const Brush *brush,
                          const int thread_id,
                          const MVert *mvert)
      : ss(ss), brush(brush), thread_id(thread_id), mvert(mvert)
  {
    init_brush_strength();
    init_brush_test();
  }

  bool paint(const Triangle &triangle, const PixelsPackage &encoded_pixels, ImBuf *image_buffer)
  {
    if (image_buffer != last_used_image_buffer_ptr) {
      last_used_image_buffer_ptr = image_buffer;
      init_brush_color(image_buffer);
    }
    image_accessor.set_image_position(image_buffer, encoded_pixels.start_image_coordinate);
    Pixel pixel = get_start_pixel(triangle, encoded_pixels);
    const Pixel add_pixel = get_delta_pixel(triangle, encoded_pixels, pixel);
    bool pixels_painted = false;
    for (int x = 0; x < encoded_pixels.num_pixels; x++) {
      if (!brush_test_fn(&test, pixel.pos)) {
        pixel += add_pixel;
        image_accessor.goto_next_pixel();
        continue;
      }

      float4 color = image_accessor.read_pixel(image_buffer);
      const float3 normal(0.0f, 0.0f, 0.0f);
      const float3 face_normal(0.0f, 0.0f, 0.0f);
      const float mask = 0.0f;
      const float falloff_strength = SCULPT_brush_strength_factor_custom_automask(
          ss,
          brush,
          pixel.pos,
          sqrtf(test.dist),
          normal,
          face_normal,
          mask,
          triangle.automasking_factor,
          thread_id);

      blend_color_interpolate_float(color, color, brush_color, falloff_strength * brush_strength);
      image_accessor.store_pixel(image_buffer, color);
      pixels_painted = true;

      image_accessor.goto_next_pixel();
      pixel += add_pixel;
    }
    return pixels_painted;
  }

 private:
  void init_brush_color(ImBuf *image_buffer)
  {
    /* TODO: use StringRefNull. */
    const char *to_colorspace = image_accessor.get_colorspace_name(image_buffer);
    if (last_used_color_space == to_colorspace) {
      return;
    }

    copy_v4_fl4(brush_color, brush->rgb[0], brush->rgb[1], brush->rgb[2], 1.0);
    /* TODO: unsure. brush color is stored in float sRGB. */
    const char *from_colorspace = IMB_colormanagement_role_colorspace_name_get(
        COLOR_ROLE_COLOR_PICKING);
    ColormanageProcessor *cm_processor = IMB_colormanagement_colorspace_processor_new(
        from_colorspace, to_colorspace);
    IMB_colormanagement_processor_apply_v4(cm_processor, brush_color);
    IMB_colormanagement_processor_free(cm_processor);
    last_used_color_space = to_colorspace;
  }

  void init_brush_strength()
  {
    brush_strength = ss->cache->bstrength;
  }
  void init_brush_test()
  {
    brush_test_fn = SCULPT_brush_test_init_with_falloff_shape(ss, &test, brush->falloff_shape);
  }

  /** Extract the staring pixel from the given encoded_pixels belonging to the triangle. */
  Pixel get_start_pixel(const Triangle &triangle, const PixelsPackage &encoded_pixels) const
  {
    return init_pixel(triangle, encoded_pixels.start_barycentric_coord);
  }

  /**
   * Extract the delta pixel that will be used to advance a Pixel instance to the next pixel. */
  Pixel get_delta_pixel(const Triangle &triangle,
                        const PixelsPackage &encoded_pixels,
                        const Pixel &start_pixel) const
  {
    Pixel result = init_pixel(
        triangle, encoded_pixels.start_barycentric_coord + triangle.add_barycentric_coord_x);
    return result - start_pixel;
  }

  Pixel init_pixel(const Triangle &triangle, const float3 weights) const
  {
    Pixel result;
    interp_v3_v3v3v3(result.pos,
                     mvert[triangle.vert_indices[0]].co,
                     mvert[triangle.vert_indices[1]].co,
                     mvert[triangle.vert_indices[2]].co,
                     weights);
    return result;
  }
};

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
    if (sculpt_brush_test_sq_fn(&test, vd.co)) {
      data->vertex_brush_tests[vd.index] = true;
    }
    data->automask_factors[vd.index] = SCULPT_automasking_factor_get(
        ss->cache->automasking, ss, vd.index);
  }
  BKE_pbvh_vertex_iter_end;
}

template<typename PaintingKernelType>
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

  /* Propagate vertex brush test to triangle. This should be extended with brush overlapping edges
   * and faces only. */
  std::vector<bool> triangle_brush_test_results(node_data->triangles.size());
  int triangle_index = 0;
  int last_poly_index = -1;
  for (Triangle &triangle : node_data->triangles) {
    for (int i = 0; i < 3; i++) {
      triangle_brush_test_results[triangle_index] =
          triangle_brush_test_results[triangle_index] ||
          data->vertex_brush_tests[triangle.vert_indices[i]];
    }
    if (last_poly_index != triangle.poly_index) {
      last_poly_index = triangle.poly_index;
      float automasking_factor = 1.0f;
      for (int t_index = triangle_index;
           t_index < node_data->triangles.size() &&
           node_data->triangles[t_index].poly_index == triangle.poly_index;
           t_index++) {
        for (int i = 0; i < 3; i++) {
          automasking_factor = min_ff(automasking_factor,
                                      data->automask_factors[triangle.vert_indices[i]]);
        }
      }

      for (int t_index = triangle_index;
           t_index < node_data->triangles.size() &&
           node_data->triangles[t_index].poly_index == triangle.poly_index;
           t_index++) {
        node_data->triangles[t_index].automasking_factor = automasking_factor;
      }
    }
    triangle_index += 1;
  }

  const int thread_id = BLI_task_parallel_thread_id(tls);
  MVert *mvert = SCULPT_mesh_deformed_mverts_get(ss);
  PaintingKernelType kernel(ss, brush, thread_id, mvert);

  int packages_clipped = 0;
  for (const PixelsPackage &encoded_pixels : node_data->encoded_pixels) {
    if (!triangle_brush_test_results[encoded_pixels.triangle_index]) {
      packages_clipped += 1;
      continue;
    }
    const Triangle &triangle = node_data->triangles[encoded_pixels.triangle_index];
    const bool pixels_painted = kernel.paint(triangle, encoded_pixels, image_buffer);

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
  data.automask_factors = Vector<float>(mesh->totvert);

  TaskParallelSettings settings;
  BKE_pbvh_parallel_range_settings(&settings, true, totnode);

  TIMEIT_START(texture_painting);
  BLI_task_parallel_range(0, totnode, &data, painting::do_vertex_brush_test, &settings);
  if (image_data.image_buffer->rect_float) {
    BLI_task_parallel_range(
        0,
        totnode,
        &data,
        painting::do_task_cb_ex<painting::PaintingKernel<painting::ImageBufferFloat4>>,
        &settings);
  }
  else {
    BLI_task_parallel_range(
        0,
        totnode,
        &data,
        painting::do_task_cb_ex<painting::PaintingKernel<painting::ImageBufferByte4>>,
        &settings);
  }
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
