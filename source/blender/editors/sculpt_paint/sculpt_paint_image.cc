/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

#include "DNA_image_types.h"
#include "DNA_material_types.h"
#include "DNA_mesh_types.h"
#include "DNA_node_types.h"
#include "DNA_object_types.h"

#include "ED_paint.h"
#include "ED_uvedit.h"

#include "BLI_math.h"
#include "BLI_math_color_blend.h"
#include "BLI_task.h"

#include "IMB_colormanagement.h"
#include "IMB_imbuf.h"

#include "BKE_image_wrappers.hh"
#include "BKE_material.h"
#include "BKE_pbvh.h"
#include "BKE_pbvh_pixels.hh"

#include "bmesh.h"

#include "NOD_shader.h"

#include "sculpt_intern.h"

namespace blender::ed::sculpt_paint::paint::image {

using namespace blender::bke::pbvh::pixels;
using namespace blender::bke::image;

struct ImageData {
  void *lock = nullptr;
  Image *image = nullptr;
  ImageUser *image_user = nullptr;

  ~ImageData()
  {
  }

  static bool init_active_image(Object *ob,
                                ImageData *r_image_data,
                                PaintModeSettings *paint_mode_settings)
  {
    return ED_paint_canvas_image_get(
        paint_mode_settings, ob, &r_image_data->image, &r_image_data->image_user);
  }
};

struct TexturePaintingUserData {
  Object *ob;
  Brush *brush;
  PBVHNode **nodes;
  ImageData image_data;
  std::vector<bool> vertex_brush_tests;
};

/** Reading and writing to image buffer with 4 float channels. */
class ImageBufferFloat4 {
 private:
  int pixel_offset;

 public:
  void set_image_position(ImBuf *image_buffer, ushort2 image_pixel_position)
  {
    pixel_offset = int(image_pixel_position.y) * image_buffer->x + int(image_pixel_position.x);
  }

  void next_pixel()
  {
    pixel_offset += 1;
  }

  float4 read_pixel(ImBuf *image_buffer) const
  {
    return &image_buffer->rect_float[pixel_offset * 4];
  }

  void write_pixel(ImBuf *image_buffer, const float4 pixel_data) const
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
  void set_image_position(ImBuf *image_buffer, ushort2 image_pixel_position)
  {
    pixel_offset = int(image_pixel_position.y) * image_buffer->x + int(image_pixel_position.x);
  }

  void next_pixel()
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

  void write_pixel(ImBuf *image_buffer, const float4 pixel_data) const
  {
    rgba_float_to_uchar(
        static_cast<uchar *>(static_cast<void *>(&image_buffer->rect[pixel_offset])), pixel_data);
  }

  const char *get_colorspace_name(ImBuf *image_buffer)
  {
    return IMB_colormanagement_get_rect_colorspace(image_buffer);
  }
};

template<typename ImageBuffer> class PaintingKernel {
  ImageBuffer image_accessor;

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

  bool paint(const Triangles &triangles, const PackedPixelRow &encoded_pixels, ImBuf *image_buffer)
  {
    if (image_buffer != last_used_image_buffer_ptr) {
      last_used_image_buffer_ptr = image_buffer;
      init_brush_color(image_buffer);
    }
    image_accessor.set_image_position(image_buffer, encoded_pixels.start_image_coordinate);
    const TrianglePaintInput triangle = triangles.get_paint_input(encoded_pixels.triangle_index);
    float3 pixel_pos = get_start_pixel_pos(triangle, encoded_pixels);
    const float3 delta_pixel_pos = get_delta_pixel_pos(triangle, encoded_pixels, pixel_pos);
    bool pixels_painted = false;
    for (int x = 0; x < encoded_pixels.num_pixels; x++) {
      if (!brush_test_fn(&test, pixel_pos)) {
        pixel_pos += delta_pixel_pos;
        image_accessor.next_pixel();
        continue;
      }

      float4 color = image_accessor.read_pixel(image_buffer);
      const float3 normal(0.0f, 0.0f, 0.0f);
      const float3 face_normal(0.0f, 0.0f, 0.0f);
      const float mask = 0.0f;
      const float falloff_strength = SCULPT_brush_strength_factor(
          ss, brush, pixel_pos, sqrtf(test.dist), normal, face_normal, mask, 0, thread_id);
      float4 paint_color = brush_color * falloff_strength * brush_strength;
      float4 buffer_color;
      blend_color_mix_float(buffer_color, color, paint_color);
      buffer_color *= brush->alpha;
      IMB_blend_color_float(color, color, buffer_color, static_cast<IMB_BlendMode>(brush->blend));
      image_accessor.write_pixel(image_buffer, color);
      pixels_painted = true;

      image_accessor.next_pixel();
      pixel_pos += delta_pixel_pos;
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

  /**
   * Extract the starting pixel position from the given encoded_pixels belonging to the triangle.
   */
  float3 get_start_pixel_pos(const TrianglePaintInput &triangle,
                             const PackedPixelRow &encoded_pixels) const
  {
    return init_pixel_pos(triangle, encoded_pixels.start_barycentric_coord);
  }

  /**
   * Extract the delta pixel position that will be used to advance a Pixel instance to the next
   * pixel.
   */
  float3 get_delta_pixel_pos(const TrianglePaintInput &triangle,
                             const PackedPixelRow &encoded_pixels,
                             const float3 &start_pixel) const
  {
    float3 result = init_pixel_pos(
        triangle, encoded_pixels.start_barycentric_coord + triangle.delta_barycentric_coord_u);
    return result - start_pixel;
  }

  float3 init_pixel_pos(const TrianglePaintInput &triangle,
                        const float3 &barycentric_weights) const
  {
    const int3 &vert_indices = triangle.vert_indices;
    float3 result;
    interp_v3_v3v3v3(result,
                     mvert[vert_indices[0]].co,
                     mvert[vert_indices[1]].co,
                     mvert[vert_indices[2]].co,
                     barycentric_weights);
    return result;
  }
};

/* Test which vertices pass the brush test and store them in the vertex_brush_tests array. */
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
  }
  BKE_pbvh_vertex_iter_end;
}

static void do_paint_pixels(void *__restrict userdata,
                            const int n,
                            const TaskParallelTLS *__restrict tls)
{
  TexturePaintingUserData *data = static_cast<TexturePaintingUserData *>(userdata);
  Object *ob = data->ob;
  SculptSession *ss = ob->sculpt;
  const Brush *brush = data->brush;
  PBVHNode *node = data->nodes[n];

  NodeData &node_data = BKE_pbvh_pixels_node_data_get(*node);

  /* Propagate vertex brush test to triangle. This should be extended with brush overlapping edges
   * and faces only. */
  /* TODO(jbakker) move to user data. to reduce reallocation. */
  std::vector<bool> triangle_brush_test_results(node_data.triangles.size());

  for (int triangle_index = 0; triangle_index < node_data.triangles.size(); triangle_index++) {
    TrianglePaintInput &triangle = node_data.triangles.get_paint_input(triangle_index);
    int3 &vert_indices = triangle.vert_indices;
    for (int i = 0; i < 3; i++) {
      triangle_brush_test_results[triangle_index] = triangle_brush_test_results[triangle_index] ||
                                                    data->vertex_brush_tests[vert_indices[i]];
    }
  }

  const int thread_id = BLI_task_parallel_thread_id(tls);
  MVert *mvert = SCULPT_mesh_deformed_mverts_get(ss);
  PaintingKernel<ImageBufferFloat4> kernel_float4(ss, brush, thread_id, mvert);
  PaintingKernel<ImageBufferByte4> kernel_byte4(ss, brush, thread_id, mvert);

  ImageUser image_user = *data->image_data.image_user;
  bool pixels_updated = false;
  for (UDIMTilePixels &tile_data : node_data.tiles) {
    LISTBASE_FOREACH (ImageTile *, tile, &data->image_data.image->tiles) {
      ImageTileWrapper image_tile(tile);
      if (image_tile.get_tile_number() == tile_data.tile_number) {
        image_user.tile = image_tile.get_tile_number();

        ImBuf *image_buffer = BKE_image_acquire_ibuf(data->image_data.image, &image_user, nullptr);
        if (image_buffer == nullptr) {
          continue;
        }

        for (const PackedPixelRow &pixel_row : tile_data.pixel_rows) {
          if (!triangle_brush_test_results[pixel_row.triangle_index]) {
            continue;
          }
          bool pixels_painted = false;
          if (image_buffer->rect_float != nullptr) {
            pixels_painted = kernel_float4.paint(node_data.triangles, pixel_row, image_buffer);
          }
          else {
            pixels_painted = kernel_byte4.paint(node_data.triangles, pixel_row, image_buffer);
          }

          if (pixels_painted) {
            tile_data.mark_dirty(pixel_row);
          }
        }

        BKE_image_release_ibuf(data->image_data.image, image_buffer, nullptr);
        pixels_updated |= tile_data.flags.dirty;
      }
      break;
    }
  }

  node_data.flags.dirty |= pixels_updated;
}

static void do_mark_dirty_regions(void *__restrict userdata,
                                  const int n,
                                  const TaskParallelTLS *__restrict UNUSED(tls))
{
  TexturePaintingUserData *data = static_cast<TexturePaintingUserData *>(userdata);
  PBVHNode *node = data->nodes[n];
  BKE_pbvh_pixels_mark_image_dirty(*node, *data->image_data.image, *data->image_data.image_user);
}

}  // namespace blender::ed::sculpt_paint::paint::image

extern "C" {

using namespace blender::ed::sculpt_paint::paint::image;

bool SCULPT_paint_image_canvas_get(PaintModeSettings *paint_mode_settings,
                                   Object *ob,
                                   Image **r_image,
                                   ImageUser **r_image_user)
{
  BLI_assert(r_image);
  BLI_assert(r_image_user);
  ImageData image_data;
  if (!ImageData::init_active_image(ob, &image_data, paint_mode_settings)) {
    return false;
  }

  *r_image = image_data.image;
  *r_image_user = image_data.image_user;
  return true;
}

bool SCULPT_use_image_paint_brush(PaintModeSettings *settings, Object *ob)
{
  if (!U.experimental.use_sculpt_texture_paint) {
    return false;
  }
  if (ob->type != OB_MESH) {
    return false;
  }
  Image *image;
  ImageUser *image_user;
  return ED_paint_canvas_image_get(settings, ob, &image, &image_user);
}

void SCULPT_do_paint_brush_image(
    PaintModeSettings *paint_mode_settings, Sculpt *sd, Object *ob, PBVHNode **nodes, int totnode)
{
  Brush *brush = BKE_paint_brush(&sd->paint);

  Mesh *mesh = (Mesh *)ob->data;

  TexturePaintingUserData data = {nullptr};
  data.ob = ob;
  data.brush = brush;
  data.nodes = nodes;
  data.vertex_brush_tests = std::vector<bool>(mesh->totvert);

  if (!ImageData::init_active_image(ob, &data.image_data, paint_mode_settings)) {
    return;
  }

  TaskParallelSettings settings;
  BKE_pbvh_parallel_range_settings(&settings, true, totnode);
  BLI_task_parallel_range(0, totnode, &data, do_vertex_brush_test, &settings);
  BLI_task_parallel_range(0, totnode, &data, do_paint_pixels, &settings);

  TaskParallelSettings settings_flush;
  BKE_pbvh_parallel_range_settings(&settings_flush, false, totnode);
  BLI_task_parallel_range(0, totnode, &data, do_mark_dirty_regions, &settings_flush);
}
}
