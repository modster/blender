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

#include "DNA_space_types.h"

#include "BKE_context.h"
#include "BKE_main.h"
#include "BKE_report.h"

#include "BLI_path_util.h"
#include "BLI_string.h"
#include "BLI_utildefines.h"

#include "BLT_translation.h"

#include "MEM_guardedalloc.h"

#include "RNA_access.h"
#include "RNA_define.h"

#include "UI_interface.h"
#include "UI_resources.h"

#include "WM_api.h"
#include "WM_types.h"

#include "DEG_depsgraph.h"

#include "IO_wavefront_obj.h"
#include "io_obj.h"

const EnumPropertyItem io_obj_transform_axis_forward[] = {
    {OBJ_AXIS_X_FORWARD, "X_FORWARD", 0, "X", "Positive X axis"},
    {OBJ_AXIS_Y_FORWARD, "Y_FORWARD", 0, "Y", "Positive Y axis"},
    {OBJ_AXIS_Z_FORWARD, "Z_FORWARD", 0, "Z", "Positive Z axis"},
    {OBJ_AXIS_NEGATIVE_X_FORWARD, "NEGATIVE_X_FORWARD", 0, "-X", "Negative X axis"},
    {OBJ_AXIS_NEGATIVE_Y_FORWARD, "NEGATIVE_Y_FORWARD", 0, "-Y (Default)", "Negative Y axis"},
    {OBJ_AXIS_NEGATIVE_Z_FORWARD, "NEGATIVE_Z_FORWARD", 0, "-Z", "Negative Z axis"},
    {0, NULL, 0, NULL, NULL}};

const EnumPropertyItem io_obj_transform_axis_up[] = {
    {OBJ_AXIS_X_UP, "X_UP", 0, "X", "Positive X axis"},
    {OBJ_AXIS_Y_UP, "Y_UP", 0, "Y", "Positive Y axis"},
    {OBJ_AXIS_Z_UP, "Z_UP", 0, "Z (Default)", "Positive Z axis"},
    {OBJ_AXIS_NEGATIVE_X_UP, "NEGATIVE_X_UP", 0, "-X", "Negative X axis"},
    {OBJ_AXIS_NEGATIVE_Y_UP, "NEGATIVE_Y_UP", 0, "-Y", "Negative Y axis"},
    {OBJ_AXIS_NEGATIVE_Z_UP, "NEGATIVE_Z_UP", 0, "-Z", "Negative Z axis"},
    {0, NULL, 0, NULL, NULL}};

const EnumPropertyItem io_obj_export_evaluation_mode[] = {
    {DAG_EVAL_RENDER,
     "DAG_EVAL_RENDER",
     0,
     "Render",
     "Modifiers need to be applied for render properties to take effect"},
    {DAG_EVAL_VIEWPORT,
     "DAG_EVAL_VIEWPORT",
     0,
     "Viewport (Default)",
     "Export objects as they appear in the viewport"},
    {0, NULL, 0, NULL, NULL}};

static int wm_obj_export_invoke(bContext *C, wmOperator *op, const wmEvent *UNUSED(event))
{
  if (!RNA_struct_property_is_set(op->ptr, "filepath")) {
    Main *bmain = CTX_data_main(C);
    char filepath[FILE_MAX];

    if (BKE_main_blendfile_path(bmain)[0] == '\0') {
      BLI_strncpy(filepath, "untitled", sizeof(filepath));
    }
    else {
      BLI_strncpy(filepath, BKE_main_blendfile_path(bmain), sizeof(filepath));
    }

    BLI_path_extension_replace(filepath, sizeof(filepath), ".obj");
    RNA_string_set(op->ptr, "filepath", filepath);
  }

  WM_event_add_fileselect(C, op);
  return OPERATOR_RUNNING_MODAL;
}

static int wm_obj_export_exec(bContext *C, wmOperator *op)
{
  if (!RNA_struct_property_is_set(op->ptr, "filepath")) {
    BKE_report(op->reports, RPT_ERROR, "No filename given");
    return OPERATOR_CANCELLED;
  }
  struct OBJExportParams export_params;
  RNA_string_get(op->ptr, "filepath", export_params.filepath);
  export_params.export_animation = RNA_boolean_get(op->ptr, "export_animation");
  export_params.start_frame = RNA_int_get(op->ptr, "start_frame");
  export_params.end_frame = RNA_int_get(op->ptr, "end_frame");

  export_params.forward_axis = RNA_enum_get(op->ptr, "forward_axis");
  export_params.up_axis = RNA_enum_get(op->ptr, "up_axis");
  export_params.scaling_factor = RNA_float_get(op->ptr, "scaling_factor");
  export_params.export_eval_mode = RNA_enum_get(op->ptr, "export_eval_mode");

  export_params.export_selected_objects = RNA_boolean_get(op->ptr, "export_selected_objects");
  export_params.export_uv = RNA_boolean_get(op->ptr, "export_uv");
  export_params.export_normals = RNA_boolean_get(op->ptr, "export_normals");
  export_params.export_materials = RNA_boolean_get(op->ptr, "export_materials");
  export_params.export_triangulated_mesh = RNA_boolean_get(op->ptr, "export_triangulated_mesh");
  export_params.export_curves_as_nurbs = RNA_boolean_get(op->ptr, "export_curves_as_nurbs");

  export_params.export_object_groups = RNA_boolean_get(op->ptr, "export_object_groups");
  export_params.export_material_groups = RNA_boolean_get(op->ptr, "export_material_groups");
  export_params.export_vertex_groups = RNA_boolean_get(op->ptr, "export_vertex_groups");
  export_params.export_smooth_groups = RNA_boolean_get(op->ptr, "export_smooth_groups");
  export_params.smooth_groups_bitflags = RNA_boolean_get(op->ptr, "smooth_group_bitflags");

  OBJ_export(C, &export_params);

  return OPERATOR_FINISHED;
}

static void ui_obj_export_settings(uiLayout *layout, PointerRNA *imfptr)
{

  const bool export_animation = RNA_boolean_get(imfptr, "export_animation");
  const bool export_smooth_groups = RNA_boolean_get(imfptr, "export_smooth_groups");

  uiLayoutSetPropSep(layout, true);
  uiLayoutSetPropDecorate(layout, false);

  /* Animation options. */
  uiLayout *box = uiLayoutBox(layout);
  uiItemL(box, IFACE_("Animation"), ICON_ANIM);
  uiLayout *col = uiLayoutColumn(box, true);
  uiLayout *sub = uiLayoutColumn(col, true);
  uiItemR(sub, imfptr, "export_animation", 0, NULL, ICON_NONE);
  sub = uiLayoutColumn(sub, true);
  uiItemR(sub, imfptr, "start_frame", 0, IFACE_("Frame Start"), ICON_NONE);
  uiItemR(sub, imfptr, "end_frame", 0, IFACE_("End"), ICON_NONE);
  uiLayoutSetEnabled(sub, export_animation);

  /* Object Transform options. */
  box = uiLayoutBox(layout);
  uiItemL(box, IFACE_("Object Transform"), ICON_OBJECT_DATA);
  col = uiLayoutColumn(box, true);
  uiItemR(col, imfptr, "forward_axis", 0, NULL, ICON_NONE);
  uiItemR(col, imfptr, "up_axis", 0, NULL, ICON_NONE);
  uiItemR(col, imfptr, "scaling_factor", 0, NULL, ICON_NONE);
  uiItemR(col, imfptr, "export_eval_mode", 0, NULL, ICON_NONE);

  /* Options for what to write. */
  box = uiLayoutBox(layout);
  uiItemL(box, IFACE_("Geometry Export Options"), ICON_EXPORT);
  col = uiLayoutColumn(box, true);
  sub = uiLayoutColumnWithHeading(col, true, IFACE_("Export"));
  uiItemR(sub, imfptr, "export_uv", 0, IFACE_("UV Coordinates"), ICON_NONE);
  uiItemR(sub, imfptr, "export_normals", 0, IFACE_("Normals"), ICON_NONE);
  uiItemR(sub, imfptr, "export_materials", 0, IFACE_("Materials"), ICON_NONE);
  uiItemR(sub, imfptr, "export_selected_objects", 0, IFACE_("Selected Objects Only"), ICON_NONE);
  uiItemR(sub, imfptr, "export_triangulated_mesh", 0, IFACE_("Triangulated Mesh"), ICON_NONE);
  uiItemR(sub, imfptr, "export_curves_as_nurbs", 0, IFACE_("Curves as NURBS"), ICON_NONE);

  box = uiLayoutBox(layout);
  uiItemL(box, IFACE_("Grouping Options"), ICON_GROUP);
  col = uiLayoutColumn(box, true);
  sub = uiLayoutColumnWithHeading(col, true, IFACE_("Export"));
  uiItemR(sub, imfptr, "export_object_groups", 0, IFACE_("Object Groups"), ICON_NONE);
  uiItemR(sub, imfptr, "export_material_groups", 0, IFACE_("Material Groups"), ICON_NONE);
  uiItemR(sub, imfptr, "export_vertex_groups", 0, IFACE_("Vertex Groups"), ICON_NONE);
  uiItemR(sub, imfptr, "export_smooth_groups", 0, IFACE_("Smooth Groups"), ICON_NONE);
  sub = uiLayoutColumn(sub, true);
  uiLayoutSetEnabled(sub, export_smooth_groups);
  uiItemR(sub, imfptr, "smooth_group_bitflags", 0, IFACE_("Smooth Group Bitflags"), ICON_NONE);
}

static void wm_obj_export_draw(bContext *UNUSED(C), wmOperator *op)
{
  PointerRNA ptr;
  RNA_pointer_create(NULL, op->type->srna, op->properties, &ptr);
  ui_obj_export_settings(op->layout, &ptr);
}

static bool wm_obj_export_check(bContext *C, wmOperator *op)
{
  char filepath[FILE_MAX] = {};
  Scene *scene = CTX_data_scene(C);
  bool ret = false;
  RNA_string_get(op->ptr, "filepath", filepath);

  if (!BLI_path_extension_check(filepath, ".obj")) {
    BLI_path_extension_ensure(filepath, FILE_MAX, ".obj");
    RNA_string_set(op->ptr, "filepath", filepath);
    ret = true;
  }

  /* Set the default export frames to the current one in viewport. */
  if (RNA_boolean_get(op->ptr, "export_animation")) {
    RNA_int_set(op->ptr, "start_frame", SFRA);
    RNA_int_set(op->ptr, "end_frame", EFRA);
  }
  else {
    RNA_int_set(op->ptr, "start_frame", CFRA);
    RNA_int_set(op->ptr, "end_frame", CFRA);
  }

  /* Both forward and up axes cannot be the same (or same except opposite sign). */
  if ((RNA_enum_get(op->ptr, "forward_axis")) % 3 == (RNA_enum_get(op->ptr, "up_axis")) % 3) {
    /* TODO (ankitm) Show a warning here. */
    RNA_enum_set(op->ptr, "up_axis", RNA_enum_get(op->ptr, "up_axis") % 3 + 1);
    ret = true;
  }

  /* One can enable smooth groups bitflags, then disable smooth groups, but smooth group bitflags
   * remain enabled. This can be confusing.
   */
  if (!RNA_boolean_get(op->ptr, "export_smooth_groups")) {
    RNA_boolean_set(op->ptr, "smooth_group_bitflags", false);
  }
  return ret;
}

void WM_OT_obj_export(struct wmOperatorType *ot)
{
  ot->name = "Export Wavefront OBJ";
  ot->description = "Save the scene to a Wavefront OBJ file";
  ot->idname = "WM_OT_obj_export";

  ot->invoke = wm_obj_export_invoke;
  ot->exec = wm_obj_export_exec;
  ot->poll = WM_operator_winactive;
  ot->ui = wm_obj_export_draw;
  ot->check = wm_obj_export_check;

  WM_operator_properties_filesel(ot,
                                 FILE_TYPE_FOLDER | FILE_TYPE_OBJECT_IO,
                                 FILE_BLENDER,
                                 FILE_SAVE,
                                 WM_FILESEL_FILEPATH | WM_FILESEL_SHOW_PROPS,
                                 FILE_DEFAULTDISPLAY,
                                 FILE_SORT_ALPHA);

  /* Animation options. */
  RNA_def_boolean(ot->srna,
                  "export_animation",
                  false,
                  "Export Animation",
                  "Export multiple frames. By default export the active frame only");
  RNA_def_int(ot->srna,
              "start_frame",
              INT_MAX,
              -INT_MAX,
              INT_MAX,
              "Start Frame",
              "The first frame to be exported",
              -INT_MAX,
              INT_MAX);
  RNA_def_int(ot->srna,
              "end_frame",
              1,
              -INT_MAX,
              INT_MAX,
              "End Frame",
              "The last frame to be exported",
              -INT_MAX,
              INT_MAX);
  /* Object transform options. */
  RNA_def_enum(ot->srna,
               "forward_axis",
               io_obj_transform_axis_forward,
               OBJ_AXIS_NEGATIVE_Y_FORWARD,
               "Forward Axis",
               "");
  RNA_def_enum(ot->srna, "up_axis", io_obj_transform_axis_up, OBJ_AXIS_Z_UP, "Up Axis", "");
  RNA_def_float(ot->srna,
                "scaling_factor",
                1.000f,
                0.001f,
                10 * 1000.000f,
                "Scale",
                "Upscale the object by this factor",
                0.01,
                1000.000f);
  /* File Writer options. */
  RNA_def_enum(ot->srna,
               "export_eval_mode",
               io_obj_export_evaluation_mode,
               DAG_EVAL_VIEWPORT,
               "Use Properties For",
               "Determines properties like object visibility, modifiers etc., where they differ "
               "for Render and Viewport");
  RNA_def_boolean(
      ot->srna,
      "export_selected_objects",
      false,
      "Export Selected Objects",
      "Export only selected objects in the scene. Exports all supported objects by default");
  RNA_def_boolean(ot->srna, "export_uv", true, "Export UVs", "");
  RNA_def_boolean(
      ot->srna,
      "export_normals",
      true,
      "Export Normals",
      "Export face normals if no face is smooth-shaded, otherwise export vertex normals");
  RNA_def_boolean(ot->srna,
                  "export_materials",
                  true,
                  "Export Materials",
                  "Export MTL library. There must be a Principled-BSDF node for image textures to "
                  "be exported to the MTL file");
  RNA_def_boolean(ot->srna,
                  "export_triangulated_mesh",
                  false,
                  "Export Triangulated Mesh",
                  "All ngons with four or more vertices will be triangulated. Meshes in "
                  "the scene will not be affected. Behaves like Triangulate Modifier with "
                  "ngon-method: \"Beauty\", quad-method: \"Shortest Diagonal\", min vertices: 4");
  RNA_def_boolean(
      ot->srna,
      "export_curves_as_nurbs",
      false,
      "Export Curves as NURBS",
      "Export curves in parametric form. If unchecked, export them as vertices and edges");

  RNA_def_boolean(ot->srna,
                  "export_object_groups",
                  false,
                  "Export Object Groups",
                  "Append mesh name to object name, separated by a '_'");
  RNA_def_boolean(ot->srna,
                  "export_material_groups",
                  false,
                  "Export Material Groups",
                  "Append mesh name and material name to object name, separated by a '_'");
  RNA_def_boolean(
      ot->srna,
      "export_vertex_groups",
      false,
      "Export Vertex Groups",
      "Write the name of the vertex group of a face. It is approximated "
      "by choosing the vertex group with the most members among the vertices of a face");
  RNA_def_boolean(ot->srna,
                  "export_smooth_groups",
                  false,
                  "Export Smooth Groups",
                  "Export smooth groups and also export per-vertex normals instead of per-face "
                  "normals, if the mesh is shaded smooth");
  RNA_def_boolean(ot->srna,
                  "smooth_group_bitflags",
                  false,
                  "Generate Bitflags for Smooth Groups",
                  "Generates upto 32 but usually much less");
}

static int wm_obj_import_invoke(bContext *C, wmOperator *op, const wmEvent *UNUSED(event))
{
  WM_event_add_fileselect(C, op);
  return OPERATOR_RUNNING_MODAL;
}

static int wm_obj_import_exec(bContext *C, wmOperator *op)
{
  if (!RNA_struct_property_is_set(op->ptr, "filepath")) {
    BKE_report(op->reports, RPT_ERROR, "No filename given");
    return OPERATOR_CANCELLED;
  }

  struct OBJImportParams import_params;
  RNA_string_get(op->ptr, "filepath", import_params.filepath);
  import_params.clamp_size = RNA_float_get(op->ptr, "clamp_size");
  import_params.forward_axis = RNA_enum_get(op->ptr, "forward_axis");
  import_params.up_axis = RNA_enum_get(op->ptr, "up_axis");

  OBJ_import(C, &import_params);

  return OPERATOR_FINISHED;
}

static void ui_obj_import_settings(uiLayout *layout, PointerRNA *imfptr)
{
  uiLayoutSetPropSep(layout, true);
  uiLayoutSetPropDecorate(layout, false);
  uiLayout *box = uiLayoutBox(layout);

  uiItemL(box, IFACE_("Transform"), ICON_OBJECT_DATA);
  uiLayout *col = uiLayoutColumn(box, false);
  uiLayout *sub = uiLayoutColumn(col, true);
  uiItemR(sub, imfptr, "clamp_size", 0, NULL, ICON_NONE);
  uiItemR(sub, imfptr, "forward_axis", 0, NULL, ICON_NONE);
  uiItemR(sub, imfptr, "up_axis", 0, NULL, ICON_NONE);
}

static void wm_obj_import_draw(bContext *C, wmOperator *op)
{
  PointerRNA ptr;
  wmWindowManager *wm = CTX_wm_manager(C);
  RNA_pointer_create(&wm->id, op->type->srna, op->properties, &ptr);
  ui_obj_import_settings(op->layout, &ptr);
}

void WM_OT_obj_import(struct wmOperatorType *ot)
{
  ot->name = "Import Wavefront OBJ";
  ot->description = "Load a Wavefront OBJ scene";
  ot->idname = "WM_OT_obj_import";

  ot->invoke = wm_obj_import_invoke;
  ot->exec = wm_obj_import_exec;
  ot->poll = WM_operator_winactive;
  ot->ui = wm_obj_import_draw;

  WM_operator_properties_filesel(ot,
                                 FILE_TYPE_FOLDER | FILE_TYPE_OBJECT_IO,
                                 FILE_BLENDER,
                                 FILE_OPENFILE,
                                 WM_FILESEL_FILEPATH | WM_FILESEL_SHOW_PROPS,
                                 FILE_DEFAULTDISPLAY,
                                 FILE_SORT_ALPHA);
  RNA_def_float(
      ot->srna,
      "clamp_size",
      0.0f,
      0.0f,
      1000.0f,
      "Clamp Bounding Box",
      "Resize the objects to keep bounding box under this value. Value 0 diables clamping",
      0.0f,
      1000.0f);
  RNA_def_enum(ot->srna,
               "forward_axis",
               io_obj_transform_axis_forward,
               OBJ_AXIS_NEGATIVE_Y_FORWARD,
               "Forward Axis",
               "");
  RNA_def_enum(ot->srna, "up_axis", io_obj_transform_axis_up, OBJ_AXIS_Z_UP, "Up Axis", "");
}
