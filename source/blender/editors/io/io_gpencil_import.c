/*
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 *
 * The Original Code is Copyright (C) 2020 Blender Foundation.
 * All rights reserved.
 */

/** \file
 * \ingroup editor/io
 */

#include <errno.h>
#include <string.h>

#include "MEM_guardedalloc.h"

#include "DNA_gpencil_types.h"
#include "DNA_space_types.h"

#include "BKE_context.h"
#include "BKE_gpencil.h"
#include "BKE_main.h"
#include "BKE_object.h"
#include "BKE_report.h"
#include "BKE_scene.h"
#include "BKE_screen.h"

#include "BLI_listbase.h"
#include "BLI_path_util.h"
#include "BLI_string.h"
#include "BLI_utildefines.h"

#include "BLT_translation.h"

#include "RNA_access.h"
#include "RNA_define.h"

#include "UI_interface.h"
#include "UI_resources.h"

#include "WM_api.h"
#include "WM_types.h"

#include "DEG_depsgraph.h"
#include "DEG_depsgraph_query.h"

#include "ED_gpencil.h"

#include "io_gpencil.h"

#include "gpencil_io.h"

/* <-------- SVG single frame import. --------> */
bool wm_gpencil_import_svg_common_check(bContext *UNUSED(C), wmOperator *op)
{

  char filepath[FILE_MAX];
  RNA_string_get(op->ptr, "filepath", filepath);

  if (!BLI_path_extension_check(filepath, ".svg")) {
    BLI_path_extension_ensure(filepath, FILE_MAX, ".svg");
    RNA_string_set(op->ptr, "filepath", filepath);
    return true;
  }

  return false;
}

static void gpencil_import_common_props(wmOperatorType *ot)
{
  PropertyRNA *prop;

  static const EnumPropertyItem target_object_modes[] = {
      {GP_TARGET_OB_NEW, "NEW", 0, "New Object", ""},
      {GP_TARGET_OB_SELECTED, "ACTIVE", 0, "Active Object", ""},
      {0, NULL, 0, NULL, NULL},
  };

  prop = RNA_def_enum(ot->srna,
                      "target",
                      target_object_modes,
                      GP_TARGET_OB_NEW,
                      "Target Object",
                      "Target grease pencil object");

  RNA_def_property_flag(prop, PROP_SKIP_SAVE);
  RNA_def_int(ot->srna,
              "resolution",
              10,
              1,
              30,
              "Resolution",
              "Resolution of the generated curves",
              1,
              20);

  RNA_def_float(ot->srna,
                "scale",
                10.0f,
                0.001f,
                100.0f,
                "Scale",
                "Scale of the final stroke",
                0.001f,
                100.0f);
}

static void ui_gpencil_import_common_settings(uiLayout *layout, PointerRNA *imfptr)
{
  uiLayout *box, *row, *col, *sub;

  box = uiLayoutBox(layout);
  row = uiLayoutRow(box, false);
  uiItemL(row, IFACE_("Import Options"), ICON_SCENE_DATA);

  col = uiLayoutColumn(box, false);

  sub = uiLayoutColumn(col, true);
  uiItemR(sub, imfptr, "target", 0, NULL, ICON_NONE);
  sub = uiLayoutColumn(col, true);
  uiItemR(sub, imfptr, "resolution", 0, NULL, ICON_NONE);
  sub = uiLayoutColumn(col, true);
  uiItemR(sub, imfptr, "scale", 0, NULL, ICON_NONE);
}

static int wm_gpencil_import_svg_invoke(bContext *C, wmOperator *op, const wmEvent *event)
{
  UNUSED_VARS(event);

  WM_event_add_fileselect(C, op);

  return OPERATOR_RUNNING_MODAL;
}

static int wm_gpencil_import_svg_exec(bContext *C, wmOperator *op)
{
  Scene *scene = CTX_data_scene(C);
  Object *ob = CTX_data_active_object(C);

  if (!RNA_struct_property_is_set(op->ptr, "filepath")) {
    BKE_report(op->reports, RPT_ERROR, "No filename given");
    return OPERATOR_CANCELLED;
  }

  /* For some reason the region cannot be retrieved from the context.
   * If a better solution is found in the future, remove this function. */
  ARegion *region = get_invoke_region(C);
  if (region == NULL) {
    BKE_report(op->reports, RPT_ERROR, "Unable to find valid 3D View area");
    return OPERATOR_CANCELLED;
  }
  View3D *v3d = get_invoke_view3d(C);

  char filename[FILE_MAX];
  RNA_string_get(op->ptr, "filepath", filename);

  /* Set flags. */
  int flag = 0;
  /* If active object is not a editable grease pencil, set to NULL to create a new object. */
  eGP_TargetObjectMode target = RNA_enum_get(op->ptr, "target");
  ob = (target == GP_TARGET_OB_SELECTED) ? CTX_data_active_object(C) : NULL;

  if (ob != NULL) {
    if (ob->type != OB_GPENCIL) {
      ob = NULL;
    }
    else if (BKE_object_obdata_is_libdata(ob)) {
      ob = NULL;
    }
  }

  const int resolution = RNA_int_get(op->ptr, "resolution");
  const float scale = RNA_float_get(op->ptr, "scale");

  struct GpencilIOParams params = {
      .C = C,
      .region = region,
      .v3d = v3d,
      .ob = ob,
      .mode = GP_IMPORT_FROM_SVG,
      .frame_start = CFRA,
      .frame_end = CFRA,
      .frame_cur = CFRA,
      .flag = flag,
      .scale = scale,
      .select_mode = 0,
      .frame_mode = 0,
      .stroke_sample = 0.0f,
      .resolution = resolution,
  };

  /* Do Import. */
  WM_cursor_wait(1);
  bool done = gpencil_io_import(filename, &params);
  WM_cursor_wait(0);

  if (done) {
    BKE_report(op->reports, RPT_INFO, "SVG file imported");
  }
  else {
    BKE_report(op->reports, RPT_WARNING, "Unable to import SVG");
  }

  return OPERATOR_FINISHED;
}

static void ui_gpencil_import_svg_settings(uiLayout *layout, PointerRNA *imfptr)
{
  uiLayout *box;

  uiLayoutSetPropSep(layout, true);
  uiLayoutSetPropDecorate(layout, false);

  box = uiLayoutBox(layout);

  ui_gpencil_import_common_settings(layout, imfptr);
}

static void wm_gpencil_import_svg_draw(bContext *UNUSED(C), wmOperator *op)
{

  PointerRNA ptr;

  RNA_pointer_create(NULL, op->type->srna, op->properties, &ptr);

  ui_gpencil_import_svg_settings(op->layout, &ptr);
}

static bool wm_gpencil_import_svg_poll(bContext *C)
{
  if (CTX_wm_window(C) == NULL) {
    return false;
  }

  return true;
}

void WM_OT_gpencil_import_svg(wmOperatorType *ot)
{
  ot->name = "Import SVG";
  ot->description = "Import SVG into grease pencil";
  ot->idname = "WM_OT_gpencil_import_svg";

  ot->invoke = wm_gpencil_import_svg_invoke;
  ot->exec = wm_gpencil_import_svg_exec;
  ot->poll = wm_gpencil_import_svg_poll;
  ot->ui = wm_gpencil_import_svg_draw;
  ot->check = wm_gpencil_import_svg_common_check;

  WM_operator_properties_filesel(ot,
                                 FILE_TYPE_OBJECT_IO,
                                 FILE_BLENDER,
                                 FILE_OPENFILE,
                                 WM_FILESEL_FILEPATH | WM_FILESEL_RELPATH | WM_FILESEL_SHOW_PROPS,
                                 FILE_DEFAULTDISPLAY,
                                 FILE_SORT_DEFAULT);

  gpencil_import_common_props(ot);
}
