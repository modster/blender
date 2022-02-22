/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

/** \file
 * \ingroup edsculpt
 * \brief 3D perspective painting.
 */

#include "MEM_guardedalloc.h"

#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"
#include "DNA_object_types.h"
#include "DNA_scene_types.h"

#include "DEG_depsgraph.h"
#include "DEG_depsgraph_query.h"

#include "BKE_context.h"
#include "BKE_mesh.h"
#include "BKE_object.h"
#include "BKE_paint.h"
#include "BKE_pbvh.h"

#include "BLI_math.h"

#include "paint_image_3d.hh"

#include "CLG_log.h"

static CLG_LogRef LOG = {"ed.sculpt_paint.image3d"};

namespace blender::ed::sculpt_paint::image3d {

static PBVH *build_pbvh_from_regular_mesh(Object *ob, Mesh *me_eval_deform, bool respect_hide)
{
  Mesh *me = BKE_object_get_original_mesh(ob);
  const int looptris_num = poly_to_tri_count(me->totpoly, me->totloop);
  PBVH *pbvh = BKE_pbvh_new();
  BKE_pbvh_respect_hide_set(pbvh, respect_hide);

  MLoopTri *looptri = static_cast<MLoopTri *>(
      MEM_malloc_arrayN(looptris_num, sizeof(MLoopTri), __func__));

  BKE_mesh_recalc_looptri(me->mloop, me->mpoly, me->mvert, me->totloop, me->totpoly, looptri);

  BKE_sculpt_sync_face_set_visibility(me, NULL);

  BKE_pbvh_build_mesh(pbvh,
                      me,
                      me->mpoly,
                      me->mloop,
                      me->mvert,
                      me->totvert,
                      &me->vdata,
                      &me->ldata,
                      &me->pdata,
                      looptri,
                      looptris_num);

  return pbvh;
}

static PBVH *pbvh_from_evaluated_object(Depsgraph *depsgraph, Object *ob)
{
  const bool respect_hide = true;
  PBVH *pbvh = nullptr;
  Object *object_eval = DEG_get_evaluated_object(depsgraph, ob);
  //  Mesh *mesh_eval = static_cast<Mesh *>(object_eval->data);
  BLI_assert_msg(ob->type == OB_MESH, "Only mesh objects are supported");
  Mesh *me_eval_deform = object_eval->runtime.mesh_deform_eval;
  pbvh = build_pbvh_from_regular_mesh(ob, me_eval_deform, respect_hide);
  return pbvh;
}

struct StrokeHandle {
  PBVH *pbvh = nullptr;
  bool owns_pbvh = false;

  virtual ~StrokeHandle()
  {
    if (pbvh && owns_pbvh) {
      BKE_pbvh_free(pbvh);
    }
    pbvh = nullptr;
    owns_pbvh = false;
  }
};

struct StrokeHandle *stroke_new(bContext *C, Object *ob)
{
  CLOG_INFO(&LOG, 2, "create new stroke");
  const Scene *scene = CTX_data_scene(C);
  Depsgraph *depsgraph = CTX_data_depsgraph_pointer(C);

  ToolSettings *ts = scene->toolsettings;
  BLI_assert_msg(ts->imapaint.mode == IMAGEPAINT_MODE_IMAGE, "Only Image mode implemented");
  Image *image = ts->imapaint.canvas;
  CLOG_INFO(&LOG, 2, " paint target image: %s", image->id.name);

  StrokeHandle *stroke_handle = MEM_new<StrokeHandle>("StrokeHandle");

  PBVH *pbvh = pbvh_from_evaluated_object(depsgraph, ob);
  stroke_handle->pbvh = pbvh;
  stroke_handle->owns_pbvh = true;

  return stroke_handle;
}

void stroke_update(struct StrokeHandle *stroke_handle, float2 prev_mouse, float2 mouse)
{
  CLOG_INFO(&LOG, 2, "new stroke step");

  
}

void stroke_free(struct StrokeHandle *stroke_handle)
{
  CLOG_INFO(&LOG, 2, "free stroke");
  MEM_delete(stroke_handle);
}

}  // namespace blender::ed::sculpt_paint::image3d