/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

/** \file
 * \ingroup edsculpt
 */

#include "DNA_material_types.h"
#include "DNA_meshdata_types.h"
#include "DNA_scene_types.h"
#include "DNA_windowmanager_types.h"

#include "BKE_context.h"
#include "BKE_image.h"
#include "BKE_material.h"
#include "BKE_mesh.h"
#include "BKE_mesh_mapping.h"

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

  VertexInput(float2 uv) : uv(uv)
  {
  }
};

class VertexShader : public AbstractVertexShader<VertexInput, int> {
 public:
  float2 image_size;
  void vertex(const VertexInputType &input, VertexOutputType *r_output) override
  {
    r_output->coord = input.uv * image_size;
  }
};

class FragmentShader : public AbstractFragmentShader<int, float4> {
 public:
  float4 color;
  void fragment(const FragmentInputType &UNUSED(input), FragmentOutputType *r_output) override
  {
    copy_v4_v4(*r_output, color);
  }
};

using RasterizerType = Rasterizer<VertexShader, FragmentShader>;

static void do_task_cb_ex(void *__restrict userdata,
                          const int n,
                          const TaskParallelTLS *__restrict UNUSED(tls))
{
  SculptThreadedTaskData *data = static_cast<SculptThreadedTaskData *>(userdata);
  SculptSession *ss = data->ob->sculpt;
  // const Brush *brush = data->brush;
  // ss->cache->bstrength;
  ImBuf *drawing_target = ss->mode.texture_paint.drawing_target;
  RasterizerType rasterizer;
  rasterizer.activate_drawing_target(drawing_target);
  rasterizer.vertex_shader().image_size = float2(drawing_target->x, drawing_target->y);
  rasterizer.fragment_shader().color = float4(1.0f, 1.0f, 1.0f, 1.0f);

  PBVHVertexIter vd;

  SculptBrushTest test;
  SculptBrushTestFn sculpt_brush_test_sq_fn = SCULPT_brush_test_init_with_falloff_shape(
      ss, &test, data->brush->falloff_shape);

  MVert *mvert = SCULPT_mesh_deformed_mverts_get(ss);

  BKE_pbvh_vertex_iter_begin (ss->pbvh, data->nodes[n], vd, PBVH_ITER_UNIQUE) {
    MeshElemMap *vert_map = &ss->pmap[vd.index];
    for (int j = 0; j < ss->pmap[vd.index].count; j++) {
      const MPoly *p = &ss->mpoly[vert_map->indices[j]];

      float poly_center[3];
      const MLoop *loopstart = &ss->mloop[p->loopstart];
      BKE_mesh_calc_poly_center(p, &ss->mloop[p->loopstart], mvert, poly_center);

      if (!sculpt_brush_test_sq_fn(&test, poly_center)) {
        continue;
      }
      if (p->totloop < 3) {
        continue;
      }

      VertexInput v1(mvert[loopstart[0].v].co);
      VertexInput v2(mvert[loopstart[1].v].co);
      VertexInput v3(mvert[loopstart[2].v].co);
      rasterizer.draw_triangle(v1, v2, v3);
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
  BKE_pbvh_parallel_range_settings(&settings, false, totnode);
  BLI_task_parallel_range(0, totnode, &data, do_task_cb_ex, &settings);

  BKE_image_release_ibuf(image, image_buffer, lock);
  ss->mode.texture_paint.drawing_target = nullptr;
}
}
}  // namespace blender::ed::sculpt_paint::texture_paint
