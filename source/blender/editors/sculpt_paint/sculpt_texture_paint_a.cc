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

class FragmentShader : public AbstractFragmentShader<float3, float4> {
 public:
  float4 color;
  const Brush *brush = nullptr;
  SculptBrushTest test;
  SculptBrushTestFn sculpt_brush_test_sq_fn;

  void fragment(const FragmentInputType &input, FragmentOutputType *r_output) override
  {
    copy_v4_v4(*r_output, color);
    float strength = sculpt_brush_test_sq_fn(&test, input) ?
                         BKE_brush_curve_strength(brush, sqrtf(test.dist), test.radius) :
                         0.0f;

    (*r_output)[3] *= strength;
  }
};

using RasterizerType = Rasterizer<VertexShader, FragmentShader, AlphaBlendMode>;

struct TexturePaintingUserData {
  Object *ob;
  Brush *brush;
  PBVHNode **nodes;
  Vector<rctf> region_to_update;
};

static void do_task_cb_ex(void *__restrict userdata,
                          const int n,
                          const TaskParallelTLS *__restrict UNUSED(tls))
{
  TexturePaintingUserData *data = static_cast<TexturePaintingUserData *>(userdata);
  Object *ob = data->ob;
  SculptSession *ss = ob->sculpt;
  const Brush *brush = data->brush;
  ImBuf *drawing_target = ss->mode.texture_paint.drawing_target;
  RasterizerType rasterizer;

  Mesh *mesh = static_cast<Mesh *>(ob->data);
  MLoopUV *ldata_uv = static_cast<MLoopUV *>(CustomData_get_layer(&mesh->ldata, CD_MLOOPUV));
  if (ldata_uv == nullptr) {
    return;
  }

  rasterizer.activate_drawing_target(drawing_target);
  rasterizer.vertex_shader().image_size = float2(drawing_target->x, drawing_target->y);
  srgb_to_linearrgb_v3_v3(rasterizer.fragment_shader().color, brush->rgb);
  FragmentShader &fragment_shader = rasterizer.fragment_shader();
  fragment_shader.color[3] = 1.0f;
  fragment_shader.brush = brush;
  fragment_shader.sculpt_brush_test_sq_fn = SCULPT_brush_test_init_with_falloff_shape(
      ss, &fragment_shader.test, brush->falloff_shape);

  PBVHVertexIter vd;

  MVert *mvert = SCULPT_mesh_deformed_mverts_get(ss);
  rctf &region_to_update = data->region_to_update[n];
  BLI_rctf_init_minmax(&region_to_update);

  BKE_pbvh_vertex_iter_begin (ss->pbvh, data->nodes[n], vd, PBVH_ITER_UNIQUE) {
    MeshElemMap *vert_map = &ss->pmap[vd.index];
    for (int j = 0; j < ss->pmap[vd.index].count; j++) {
      const MPoly *p = &ss->mpoly[vert_map->indices[j]];
      if (p->totloop < 3) {
        continue;
      }

      float poly_center[3];
      const MLoop *loopstart = &ss->mloop[p->loopstart];
      BKE_mesh_calc_poly_center(p, &ss->mloop[p->loopstart], mvert, poly_center);
      if (!fragment_shader.sculpt_brush_test_sq_fn(&fragment_shader.test, poly_center)) {
        continue;
      }

      for (int triangle = 0; triangle < p->totloop - 2; triangle++) {
        const int v1_index = loopstart[0].v;
        const int v2_index = loopstart[triangle + 1].v;
        const int v3_index = loopstart[triangle + 2].v;
        const int v1_loop_index = p->loopstart;
        const int v2_loop_index = p->loopstart + triangle + 1;
        const int v3_loop_index = p->loopstart + triangle + 2;

        VertexInput v1(mvert[v1_index].co, ldata_uv[v1_loop_index].uv);
        VertexInput v2(mvert[v2_index].co, ldata_uv[v2_loop_index].uv);
        VertexInput v3(mvert[v3_index].co, ldata_uv[v3_loop_index].uv);
        rasterizer.draw_triangle(v1, v2, v3);

        BLI_rctf_do_minmax_v(&region_to_update, v1.uv);
        BLI_rctf_do_minmax_v(&region_to_update, v2.uv);
        BLI_rctf_do_minmax_v(&region_to_update, v3.uv);
      }
    }
  }
  BKE_pbvh_vertex_iter_end;

  rasterizer.deactivate_drawing_target();
}

extern "C" {
void SCULPT_do_texture_paint_brush(Sculpt *sd, Object *ob, PBVHNode **nodes, int totnode)
{
  SculptSession *ss = ob->sculpt;
  Brush *brush = BKE_paint_brush(&sd->paint);

  void *lock;
  Image *image;
  ImageUser *image_user;

  ED_object_get_active_image(ob, 1, &image, &image_user, nullptr, nullptr);
  if (image == nullptr) {
    return;
  }
  ImBuf *image_buffer = BKE_image_acquire_ibuf(image, image_user, &lock);
  if (image_buffer == nullptr) {
    return;
  }
  ss->mode.texture_paint.drawing_target = image_buffer;

  TexturePaintingUserData data = {nullptr};
  data.ob = ob;
  data.brush = brush;
  data.nodes = nodes;
  data.region_to_update.resize(totnode);

  TaskParallelSettings settings;
  BKE_pbvh_parallel_range_settings(&settings, true, totnode);

  TIMEIT_START(texture_painting);
  BLI_task_parallel_range(0, totnode, &data, do_task_cb_ex, &settings);
  TIMEIT_END(texture_painting);

  for (int i = 0; i < totnode; i++) {
    rcti region_to_update;
    region_to_update.xmin = data.region_to_update[i].xmin * image_buffer->x;
    region_to_update.xmax = data.region_to_update[i].xmax * image_buffer->x;
    region_to_update.ymin = data.region_to_update[i].ymin * image_buffer->y;
    region_to_update.ymax = data.region_to_update[i].ymax * image_buffer->y;

    /* TODO: Tiled images. */
    BKE_image_partial_update_mark_region(
        image, static_cast<ImageTile *>(image->tiles.first), image_buffer, &region_to_update);
  }

  BKE_image_release_ibuf(image, image_buffer, lock);
  ss->mode.texture_paint.drawing_target = nullptr;
}
}
}  // namespace blender::ed::sculpt_paint::texture_paint
