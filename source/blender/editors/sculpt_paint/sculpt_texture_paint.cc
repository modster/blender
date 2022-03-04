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

#include "BLI_task.h"

#include "IMB_rasterizer.hh"

#include "WM_types.h"

#include "bmesh.h"

#include "ED_uvedit.h"

#include "sculpt_intern.h"

namespace blender::ed::sculpt_paint::texture_paint {

using namespace imbuf::rasterizer;

struct VertexInput {
  float2 uv;
  float strength;

  VertexInput(float2 uv, float strength) : uv(uv), strength(strength)
  {
  }
};

class VertexShader : public AbstractVertexShader<VertexInput, float> {
 public:
  float2 image_size;
  void vertex(const VertexInputType &input, VertexOutputType *r_output) override
  {
    r_output->coord = input.uv * image_size;
    r_output->data = input.strength;
  }
};

class FragmentShader : public AbstractFragmentShader<float, float4> {
 public:
  float4 color;
  void fragment(const FragmentInputType &input, FragmentOutputType *r_output) override
  {
    copy_v4_v4(*r_output, color);
    (*r_output)[3] = input;
  }
};

using RasterizerType = Rasterizer<VertexShader, FragmentShader, AlphaBlendMode>;

static void do_task_cb_ex(void *__restrict userdata,
                          const int n,
                          const TaskParallelTLS *__restrict UNUSED(tls))
{
  SculptThreadedTaskData *data = static_cast<SculptThreadedTaskData *>(userdata);
  Object *ob = data->ob;
  SculptSession *ss = ob->sculpt;
  StrokeCache *cache = ss->cache;
  const Brush *brush = data->brush;
  // ss->cache->bstrength;
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
  rasterizer.fragment_shader().color[3] = 1.0;

  PBVHVertexIter vd;

  SculptBrushTest test;
  SculptBrushTestFn sculpt_brush_test_sq_fn = SCULPT_brush_test_init_with_falloff_shape(
      ss, &test, data->brush->falloff_shape);

  MVert *mvert = SCULPT_mesh_deformed_mverts_get(ss);

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

      if (!sculpt_brush_test_sq_fn(&test, poly_center)) {
        continue;
      }
      const float strength = BKE_brush_curve_strength(brush, sqrtf(test.dist), cache->radius);

      for (int triangle = 0; triangle < p->totloop - 2; triangle++) {
        const int v1_index = p->loopstart;                 // loopstart[0].v;
        const int v2_index = p->loopstart + triangle + 1;  // loopstart[triangle + 1].v;
        const int v3_index = p->loopstart + triangle + 2;  // loopstart[triangle + 2].v;
        VertexInput v1(ldata_uv[v1_index].uv, strength);
        VertexInput v2(ldata_uv[v2_index].uv, strength);
        VertexInput v3(ldata_uv[v3_index].uv, strength);
        rasterizer.draw_triangle(v1, v2, v3);
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

  SculptThreadedTaskData data = {nullptr};
  data.sd = sd;
  data.ob = ob;
  data.brush = brush;
  data.nodes = nodes;

  TaskParallelSettings settings;
  BKE_pbvh_parallel_range_settings(&settings, true, totnode);
  BLI_task_parallel_range(0, totnode, &data, do_task_cb_ex, &settings);

  BKE_image_release_ibuf(image, image_buffer, lock);
  // TODO(do partial update
  BKE_image_partial_update_mark_full_update(image);
  ss->mode.texture_paint.drawing_target = nullptr;
}
}
}  // namespace blender::ed::sculpt_paint::texture_paint
