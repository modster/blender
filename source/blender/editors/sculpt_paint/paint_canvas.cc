/* SPDX-License-Identifier: GPL-2.0-or-later */

#include <optional>

#include "ED_paint.h"

#include "BLI_string_ref.hh"
#include "BLI_vector.hh"

#include "DNA_image_types.h"
#include "DNA_material_types.h"
#include "DNA_mesh_types.h"
#include "DNA_node_types.h"
#include "DNA_workspace_types.h"

#include "BKE_context.h"
#include "BKE_customdata.h"
#include "BKE_material.h"
#include "BKE_paint.h"

#include "DEG_depsgraph.h"

#include "NOD_shader.h"

#include "UI_resources.h"

#include "RNA_access.h"
#include "RNA_define.h"

#include "WM_toolsystem.h"

namespace blender::ed::sculpt_paint::canvas {
static TexPaintSlot *get_active_slot(Object *ob)
{
  Material *mat = BKE_object_material_get(ob, ob->actcol);
  if (mat == nullptr) {
    return nullptr;
  }
  if (mat->texpaintslot == nullptr) {
    return nullptr;
  }
  if (mat->paint_active_slot >= mat->tot_slots) {
    return nullptr;
  }

  TexPaintSlot *slot = &mat->texpaintslot[mat->paint_active_slot];
  return slot;
}

}  // namespace blender::ed::sculpt_paint::canvas

extern "C" {

using namespace blender;
using namespace blender::ed::sculpt_paint::canvas;

eV3DShadingColorType ED_paint_draw_color_override(bContext *C,
                                                  const PaintModeSettings *settings,
                                                  Object *ob,
                                                  eV3DShadingColorType orig_color_type)
{
  if (!ED_paint_tool_use_canvas(C, ob)) {
    return orig_color_type;
  }

  eV3DShadingColorType override = orig_color_type;
  switch (settings->canvas_source) {
    case PAINT_CANVAS_SOURCE_COLOR_ATTRIBUTE:
      override = V3D_SHADING_VERTEX_COLOR;
      break;
    case PAINT_CANVAS_SOURCE_IMAGE:
      override = V3D_SHADING_TEXTURE_COLOR;
      break;
    case PAINT_CANVAS_SOURCE_MATERIAL: {
      TexPaintSlot *slot = get_active_slot(ob);
      if (slot == nullptr) {
        break;
      }

      if (slot->ima) {
        override = V3D_SHADING_TEXTURE_COLOR;
      }
      if (slot->attribute_name) {
        override = V3D_SHADING_VERTEX_COLOR;
      }

      break;
    }
  }

  /* Reset to original color based on enabled experimental features */
  if (!U.experimental.use_sculpt_vertex_colors && override == V3D_SHADING_VERTEX_COLOR) {
    return orig_color_type;
  }
  if (!U.experimental.use_sculpt_texture_paint && override == V3D_SHADING_TEXTURE_COLOR) {
    return orig_color_type;
  }

  return override;
}

Image *ED_paint_canvas_image_get(const struct PaintModeSettings *settings, struct Object *ob)
{
  switch (settings->canvas_source) {
    case PAINT_CANVAS_SOURCE_COLOR_ATTRIBUTE:
      return nullptr;
    case PAINT_CANVAS_SOURCE_IMAGE:
      return settings->image;
    case PAINT_CANVAS_SOURCE_MATERIAL: {
      TexPaintSlot *slot = get_active_slot(ob);
      if (slot == nullptr) {
        break;
      }
      return slot->ima;
    }
  }
  return nullptr;
}

int ED_paint_canvas_uvmap_layer_index_get(const struct PaintModeSettings *settings,
                                          struct Object *ob)
{
  switch (settings->canvas_source) {
    case PAINT_CANVAS_SOURCE_COLOR_ATTRIBUTE:
      return -1;
    case PAINT_CANVAS_SOURCE_IMAGE: {
      /* Use active uv map of the object. */
      if (ob->type != OB_MESH) {
        return -1;
      }

      const Mesh *mesh = static_cast<Mesh *>(ob->data);
      return CustomData_get_active_layer_index(&mesh->ldata, CD_MLOOPUV);
    }
    case PAINT_CANVAS_SOURCE_MATERIAL: {
      /* Use uv map of the canvas. */
      TexPaintSlot *slot = get_active_slot(ob);
      if (slot == nullptr) {
        break;
      }

      if (ob->type != OB_MESH) {
        return -1;
      }

      if (slot->uvname == nullptr) {
        return -1;
      }

      const Mesh *mesh = static_cast<Mesh *>(ob->data);
      return CustomData_get_named_layer_index(&mesh->ldata, CD_MLOOPUV, slot->uvname);
    }
  }
  return -1;
}

bool ED_paint_tool_use_canvas(struct bContext *C, struct Object *ob)
{
  /* Quick exit, only sculpt tools can use canvas. */
  if (ob == nullptr || ob->sculpt == nullptr) {
    return false;
  }

  bToolRef *tref = WM_toolsystem_ref_from_context(C);
  if (tref != nullptr) {
    if (STREQ(tref->idname, "builtin_brush.Paint")) {
      return true;
    }
    if (STREQ(tref->idname, "builtin.color_filter")) {
      return true;
    }
  }

  return false;
}

void ED_paint_do_msg_notify_active_tool_changed(struct bContext *C,
                                                struct wmMsgSubscribeKey *msg_key,
                                                struct wmMsgSubscribeValue *msg_val)
{
  Object *ob = CTX_data_active_object(C);
  if (ob == nullptr) {
    return;
  }
  DEG_id_tag_update(&ob->id, ID_RECALC_SHADING);
}
}