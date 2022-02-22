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

  if (ob->sculpt == nullptr) {
    BKE_object_sculpt_data_create(ob);
  }
  PBVH *pbvh = BKE_sculpt_object_pbvh_ensure(depsgraph, ob);
  BLI_assert_msg(pbvh != nullptr, "Unable to retrieve PBVH from sculptsession");

  StrokeHandle *stroke_handle = MEM_new<StrokeHandle>("StrokeHandle");
  stroke_handle->pbvh = pbvh;
  stroke_handle->owns_pbvh = false;

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