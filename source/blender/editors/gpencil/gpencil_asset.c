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
 * The Original Code is Copyright (C) 2021, Blender Foundation
 * This is a new part of Blender
 * Operators for editing Grease Pencil strokes
 */

/** \file
 * \ingroup edgpencil
 */

#include "BLI_blenlib.h"
#include "BLI_ghash.h"
#include "BLI_math.h"
#include "BLI_utildefines.h"

#include "BLT_translation.h"

#include "MEM_guardedalloc.h"

#include "DNA_gpencil_types.h"

#include "BKE_context.h"
#include "BKE_gpencil.h"
#include "BKE_gpencil_geom.h"
#include "BKE_lib_id.h"
#include "BKE_main.h"
#include "BKE_object.h"
#include "BKE_report.h"

#include "UI_interface.h"
#include "UI_resources.h"

#include "WM_api.h"
#include "WM_types.h"

#include "RNA_access.h"
#include "RNA_define.h"
#include "RNA_enum_types.h"

#include "GPU_framebuffer.h"
#include "GPU_immediate.h"
#include "GPU_matrix.h"
#include "GPU_state.h"

#include "ED_asset.h"
#include "ED_gpencil.h"
#include "ED_screen.h"
#include "ED_space_api.h"

#include "DEG_depsgraph.h"
#include "DEG_depsgraph_build.h"

#include "gpencil_intern.h"

/* Temporary Asset import operation data */
typedef struct tGPDasset {
  /** Current depsgraph from context. */
  struct Depsgraph *depsgraph;
  /** current scene from context. */
  struct Scene *scene;
  /** area where painting originated. */
  struct ScrArea *area;
  /** region where painting originated. */
  struct ARegion *region;
  /** Current object. */
  struct Object *ob;
  /** Current GP datablock. */
  struct bGPdata *gpd;
  /** Asset GP datablock. */
  struct bGPdata *gpd_asset;
  /* Space Conversion Data */
  struct GP_SpaceConversion gsc;

  /** Current frame number. */
  int cframe;
  /** General Flag. */
  int flag;
  /** Transform mode. */
  short mode;

  /** Drop initial position. */
  int drop_x, drop_y;
  /** Mouse last click position. */
  int mouse[2];
  /** 2D Cage vertices. */
  rctf rect_cage;
  /** 2D cage manipulator points *
   *   0----1----2
   *   |         |
   *   7         3
   *   |         |
   *   6----5----4
   */
  float manipulator[8][2];

  /** Hash of new created layers. */
  struct GHash *asset_layers;
  /** Hash of new created frames. */
  struct GHash *asset_frames;
  /** Hash of new created strokes. */
  struct GHash *asset_strokes;

  /** Handle for drawing while operator is running. */
  void *draw_handle_3d;

} tGPDasset;

typedef enum eGP_AssetTransformMode {
  /* Location. */
  GP_ASSET_TRANSFORM_LOC = 0,
  /* Rotation. */
  GP_ASSET_TRANSFORM_ROT = 1,
  /* Scale. */
  GP_ASSET_TRANSFORM_SCALE = 2,
} eGP_AssetTransformMode;

static bool gpencil_asset_generic_poll(bContext *C)
{
  if (U.experimental.use_asset_browser == false) {
    return false;
  }

  Object *ob = CTX_data_active_object(C);
  if ((ob == NULL) || (ob->type != OB_GPENCIL)) {
    return false;
  }

  return ED_operator_view3d_active(C);
}

/* -------------------------------------------------------------------- */
/** \name Create Grease Pencil Asset operator
 * \{ */

typedef enum eGP_AssetModes {
  /* Active Layer. */
  GP_ASSET_MODE_LAYER = 0,
  /* Active Frame. */
  GP_ASSET_MODE_FRAME,
  /* Active Frame All Layers. */
  GP_ASSET_MODE_FRAME_ALL_LAYERS,
  /* Selected Strokesd. */
  GP_ASSET_MODE_SELECTED_STROKES,
} eGP_AssetModes;

static int gpencil_asset_create_exec(bContext *C, wmOperator *op)
{
  Main *bmain = CTX_data_main(C);
  Object *ob = CTX_data_active_object(C);
  bGPdata *gpd_src = ob->data;

  eGP_AssetModes mode = RNA_enum_get(op->ptr, "mode");

  /* Create a copy of selected datablock. */
  bGPdata *gpd = (bGPdata *)BKE_id_copy(bmain, &gpd_src->id);
  /* Enable fake user by default. */
  id_fake_user_set(&gpd->id);
  /* Disable Edit mode. */
  gpd->flag &= ~GP_DATA_STROKE_EDITMODE;

  bGPDlayer *gpl_active = BKE_gpencil_layer_active_get(gpd);

  LISTBASE_FOREACH_MUTABLE (bGPDlayer *, gpl, &gpd->layers) {
    /* If Layer o Active Frame mode, delete non active layers. */
    if ((ELEM(mode, GP_ASSET_MODE_LAYER, GP_ASSET_MODE_FRAME)) && (gpl != gpl_active)) {
      BKE_gpencil_layer_delete(gpd, gpl);
      continue;
    }

    bGPDframe *gpf_active = gpl->actframe;

    LISTBASE_FOREACH_MUTABLE (bGPDframe *, gpf, &gpl->frames) {
      /* If Active Frame mode, delete non active frames. */
      if ((ELEM(mode, GP_ASSET_MODE_FRAME, GP_ASSET_MODE_FRAME_ALL_LAYERS)) &&
          (gpf != gpf_active)) {
        BKE_gpencil_layer_frame_delete(gpl, gpf);
        continue;
      }
      /* Remove any unselected stroke if SELECTED mode. */
      if (mode == GP_ASSET_MODE_SELECTED_STROKES) {
        LISTBASE_FOREACH_MUTABLE (bGPDstroke *, gps, &gpf->strokes) {
          if ((gps->flag & GP_STROKE_SELECT) == 0) {
            BLI_remlink(&gpf->strokes, gps);
            BKE_gpencil_free_stroke(gps);
            continue;
          }
        }
      }
    }
  }

  if (ED_asset_mark_id(C, &gpd->id)) {
  }

  WM_main_add_notifier(NC_ID | NA_EDITED, NULL);
  WM_main_add_notifier(NC_ASSET | NA_ADDED, NULL);

  return OPERATOR_FINISHED;
}

void GPENCIL_OT_asset_create(wmOperatorType *ot)
{
  static const EnumPropertyItem mode_types[] = {
      {GP_ASSET_MODE_LAYER, "LAYER", 0, "Active Layer", ""},
      {GP_ASSET_MODE_FRAME, "FRAME", 0, "Active Frame (Active Layer)", ""},
      {GP_ASSET_MODE_FRAME_ALL_LAYERS, "FRAME_ALL", 0, "Active Frame (All Layers)", ""},
      {GP_ASSET_MODE_SELECTED_STROKES, "SELECTED", 0, "Selected Strokes", ""},
      {0, NULL, 0, NULL, NULL},
  };

  /* identifiers */
  ot->name = "Create Grease Pencil Asset";
  ot->idname = "GPENCIL_OT_asset_create";
  ot->description = "Create asset from sections of the active object";

  /* callbacks */
  ot->invoke = WM_menu_invoke;
  ot->exec = gpencil_asset_create_exec;
  ot->poll = gpencil_asset_generic_poll;

  /* flags */
  ot->flag = OPTYPE_REGISTER | OPTYPE_UNDO;

  /* properties */
  ot->prop = RNA_def_enum(
      ot->srna, "mode", mode_types, GP_ASSET_MODE_SELECTED_STROKES, "Mode", "");
}

/** \} */
/* -------------------------------------------------------------------- */
/** \name Import Grease Pencil Asset into existing datablock operator
 * \{ */

/* Helper: Update all imported strokes */
static void gpencil_asset_import_update_strokes(bContext *C, tGPDasset *tgpa)
{
  bGPdata *gpd = tgpa->gpd;

  // TODO
  DEG_id_tag_update(&gpd->id, ID_RECALC_TRANSFORM | ID_RECALC_GEOMETRY);
  WM_event_add_notifier(C, NC_GPENCIL | NA_EDITED, NULL);
}

/* Helper: Draw status message while the user is running the operator */
static void gpencil_asset_import_status_indicators(bContext *C, tGPDasset *tgpa)
{
  char status_str[UI_MAX_DRAW_STR];
  char msg_str[UI_MAX_DRAW_STR];
  bGPdata *gpd_asset = tgpa->gpd_asset;
  const char *mode_txt[] = {"Location", "Rotation", "Scale"};

  BLI_strncpy(msg_str, TIP_("Importing Asset"), UI_MAX_DRAW_STR);

  BLI_snprintf(status_str,
               sizeof(status_str),
               "%s %s (%s)",
               msg_str,
               gpd_asset->id.name + 2,
               mode_txt[tgpa->mode]);

  ED_area_status_text(tgpa->area, status_str);
  ED_workspace_status_text(C,
                           TIP_("ESC/RMB to cancel, Enter to confirm, LMB to Move, "
                                "Shift+LMB to Rotate, Wheelmouse to Scale "));
}

/* Update screen and stroke */
static void gpencil_asset_import_update(bContext *C, wmOperator *op, tGPDasset *tgpa)
{
  /* update shift indicator in header */
  gpencil_asset_import_status_indicators(C, tgpa);
  /* update points position */
  gpencil_asset_import_update_strokes(C, tgpa);
}

/* ----------------------- */

/* Exit and free memory */
static void gpencil_asset_import_exit(bContext *C, wmOperator *op)
{
  tGPDasset *tgpa = op->customdata;
  bGPdata *gpd = tgpa->gpd;

  /* don't assume that operator data exists at all */
  if (tgpa) {
    /* clear status message area */
    ED_area_status_text(tgpa->area, NULL);
    ED_workspace_status_text(C, NULL);

    /* Clear any temp stroke. */
    // TODO

    /* Free Hash tablets. */
    if (tgpa->asset_layers != NULL) {
      BLI_ghash_free(tgpa->asset_layers, NULL, NULL);
    }
    if (tgpa->asset_frames != NULL) {
      BLI_ghash_free(tgpa->asset_frames, NULL, NULL);
    }
    if (tgpa->asset_strokes != NULL) {
      BLI_ghash_free(tgpa->asset_strokes, NULL, NULL);
    }

    /* Remove drawing handler. */
    if (tgpa->draw_handle_3d) {
      ED_region_draw_cb_exit(tgpa->region->type, tgpa->draw_handle_3d);
    }

    MEM_SAFE_FREE(tgpa);
  }
  DEG_id_tag_update(&gpd->id, ID_RECALC_TRANSFORM | ID_RECALC_GEOMETRY);
  WM_event_add_notifier(C, NC_GPENCIL | NA_EDITED, NULL);

  /* clear pointer */
  op->customdata = NULL;
}

/* Init new temporary interpolation data */
static bool gpencil_asset_import_set_init_values(bContext *C,
                                                 wmOperator *op,
                                                 ID *id,
                                                 tGPDasset *tgpa)
{
  /* Save current settings. */
  tgpa->depsgraph = CTX_data_ensure_evaluated_depsgraph(C);
  tgpa->scene = CTX_data_scene(C);
  tgpa->area = CTX_wm_area(C);
  tgpa->region = CTX_wm_region(C);
  tgpa->ob = CTX_data_active_object(C);

  /* Setup space conversions data. */
  gpencil_point_conversion_init(C, &tgpa->gsc);

  /* Save current frame number. */
  tgpa->cframe = tgpa->scene->r.cfra;

  /* Target GP datablock. */
  tgpa->gpd = tgpa->ob->data;
  /* Asset GP datablock. */
  tgpa->gpd_asset = (bGPdata *)id;

  tgpa->mode = GP_ASSET_TRANSFORM_LOC;

  tgpa->asset_layers = NULL;
  tgpa->asset_frames = NULL;
  tgpa->asset_strokes = NULL;

  return true;
}

/* Allocate memory and initialize values */
static tGPDasset *gpencil_session_init_asset_import(bContext *C, wmOperator *op)
{
  Main *bmain = CTX_data_main(C);
  ID *id = NULL;

  PropertyRNA *prop_name = RNA_struct_find_property(op->ptr, "name");
  PropertyRNA *prop_type = RNA_struct_find_property(op->ptr, "type");

  /* These shouldn't fail when created by outliner dropping as it checks the ID is valid. */
  if (!RNA_property_is_set(op->ptr, prop_name) || !RNA_property_is_set(op->ptr, prop_type)) {
    return NULL;
  }
  const short id_type = RNA_property_enum_get(op->ptr, prop_type);
  char name[MAX_ID_NAME - 2];
  RNA_property_string_get(op->ptr, prop_name, name);
  id = BKE_libblock_find_name(bmain, id_type, name);
  if (id == NULL) {
    return NULL;
  }
  const int object_type = BKE_object_obdata_to_type(id);
  if (object_type == -1) {
    return NULL;
  }

  tGPDasset *tgpa = MEM_callocN(sizeof(tGPDasset), "GPencil Asset Import Data");

  /* Save initial values. */
  gpencil_asset_import_set_init_values(C, op, id, tgpa);

  /* return context data for running operator */
  return tgpa;
}

/* Init interpolation: Allocate memory and set init values */
static int gpencil_asset_import_init(bContext *C, wmOperator *op)
{
  tGPDasset *tgpa;

  /* check context */
  tgpa = op->customdata = gpencil_session_init_asset_import(C, op);
  if (tgpa == NULL) {
    /* something wasn't set correctly in context */
    gpencil_asset_import_exit(C, op);
    return 0;
  }

  /* everything is now setup ok */
  return 1;
}

/* Helper: Compute 2D cage size in screen pixels. */
static void gpencil_2d_cage_prepare(tGPDasset *tgpa)
{
  /* Add some oversize. */
  const float oversize = 5.0f;

  float diff_mat[4][4];
  unit_m4(diff_mat);

  float cage_min[2];
  float cage_max[2];
  INIT_MINMAX2(cage_min, cage_max);

  GHashIterator gh_iter;
  GHASH_ITER (gh_iter, tgpa->asset_strokes) {
    // TODO: All strokes or only active frame?
    bGPDstroke *gps = (bGPDstroke *)BLI_ghashIterator_getKey(&gh_iter);
    if (is_zero_v3(gps->boundbox_min)) {
      BKE_gpencil_stroke_boundingbox_calc(gps);
    }
    float boundbox_min[2];
    float boundbox_max[2];
    ED_gpencil_projected_2d_bound_box(&tgpa->gsc, gps, diff_mat, boundbox_min, boundbox_max);
    minmax_v2v2_v2(cage_min, cage_max, boundbox_min);
    minmax_v2v2_v2(cage_min, cage_max, boundbox_max);
  }

  tgpa->rect_cage.xmin = cage_min[0] - oversize;
  tgpa->rect_cage.ymin = cage_min[1] - oversize;
  tgpa->rect_cage.xmax = cage_max[0] + oversize;
  tgpa->rect_cage.ymax = cage_max[1] + oversize;

  /* Manipulator points */
  tgpa->manipulator[0][0] = tgpa->rect_cage.xmin;
  tgpa->manipulator[0][1] = tgpa->rect_cage.ymax;

  tgpa->manipulator[1][0] = tgpa->rect_cage.xmin +
                            ((tgpa->rect_cage.xmax - tgpa->rect_cage.xmin) * 0.5f);
  tgpa->manipulator[1][1] = tgpa->rect_cage.ymax;

  tgpa->manipulator[2][0] = tgpa->rect_cage.xmax;
  tgpa->manipulator[2][1] = tgpa->rect_cage.ymax;

  tgpa->manipulator[3][0] = tgpa->rect_cage.xmax;
  tgpa->manipulator[3][1] = tgpa->rect_cage.ymin +
                            ((tgpa->rect_cage.ymax - tgpa->rect_cage.ymin) * 0.5f);

  tgpa->manipulator[4][0] = tgpa->rect_cage.xmax;
  tgpa->manipulator[4][1] = tgpa->rect_cage.ymin;

  tgpa->manipulator[5][0] = tgpa->rect_cage.xmin +
                            ((tgpa->rect_cage.xmax - tgpa->rect_cage.xmin) * 0.5f);
  tgpa->manipulator[5][1] = tgpa->rect_cage.ymin;

  tgpa->manipulator[6][0] = tgpa->rect_cage.xmin;
  tgpa->manipulator[6][1] = tgpa->rect_cage.ymin;

  tgpa->manipulator[7][0] = tgpa->rect_cage.xmin;
  tgpa->manipulator[7][1] = tgpa->rect_cage.ymin +
                            ((tgpa->rect_cage.ymax - tgpa->rect_cage.ymin) * 0.5f);
}

/* Helper: Load all strokes in the target datablock. */
static void gpencil_asset_add_strokes(tGPDasset *tgpa)
{
  bGPdata *gpd_target = tgpa->gpd;
  bGPdata *gpd_asset = tgpa->gpd_asset;

  LISTBASE_FOREACH (bGPDlayer *, gpl_asset, &gpd_asset->layers) {
    /* Check if Layer is in target datablock. */
    bGPDlayer *gpl_target = BKE_gpencil_layer_get_by_name(gpd_target, gpl_asset->info, false);
    if (gpl_target == NULL) {
      gpl_target = BKE_gpencil_layer_addnew(gpd_target, gpl_asset->info, false, false);
      BLI_assert(gpl_target != NULL);

      if (tgpa->asset_layers == NULL) {
        tgpa->asset_layers = BLI_ghash_ptr_new(__func__);
      }
      /* Add to the hash to remove if operator is canceled. */
      BLI_ghash_insert(tgpa->asset_layers, gpl_target, gpl_target);
    }

    LISTBASE_FOREACH (bGPDframe *, gpf_asset, &gpl_asset->frames) {
      /* Check if frame is in target layer. */
      bGPDframe *gpf_target = BKE_gpencil_layer_frame_get(
          gpl_target, gpf_asset->framenum, GP_GETFRAME_USE_PREV);
      if (gpf_target == NULL) {
        gpf_target = BKE_gpencil_layer_frame_get(
            gpl_target, gpf_asset->framenum, GP_GETFRAME_ADD_NEW);
        BLI_assert(gpf_target != NULL);

        if (tgpa->asset_frames == NULL) {
          tgpa->asset_frames = BLI_ghash_ptr_new(__func__);
        }
        /* Add to the hash to remove if operator is canceled. */
        if (!BLI_ghash_haskey(tgpa->asset_frames, gpf_target)) {
          /* Add the hash key with a reference to the layer. */
          BLI_ghash_insert(tgpa->asset_frames, gpf_target, gpl_target);
        }
      }
      /* Loop all strokes and duplicate. */
      if (tgpa->asset_strokes == NULL) {
        tgpa->asset_strokes = BLI_ghash_ptr_new(__func__);
      }

      LISTBASE_FOREACH (bGPDstroke *, gps_asset, &gpf_asset->strokes) {
        bGPDstroke *gps_target = BKE_gpencil_stroke_duplicate(gps_asset, true, true);
        BLI_addtail(&gpf_target->strokes, gps_target);
        /* Add the hash key with a reference to the frame. */
        BLI_ghash_insert(tgpa->asset_strokes, gps_target, gpf_target);
      }
    }
  }
  /* Prepare 2D cage. */
  gpencil_2d_cage_prepare(tgpa);
}

/* Draw a cage for manipulate asset */
static void gpencil_draw_cage(tGPDasset *tgpa)
{
  GPUVertFormat *format = immVertexFormat();
  uint pos = GPU_vertformat_attr_add(format, "pos", GPU_COMP_F32, 2, GPU_FETCH_FLOAT);

  GPU_blend(GPU_BLEND_ALPHA);

  /* Draw dash box. */
  immBindBuiltinProgram(GPU_SHADER_2D_LINE_DASHED_UNIFORM_COLOR);
  GPU_line_width(1.0f);

  float viewport_size[4];
  GPU_viewport_size_get_f(viewport_size);
  immUniform2f("viewport_size", viewport_size[2], viewport_size[3]);

  immUniform1i("colors_len", 0); /* "simple" mode */
  immUniform1f("dash_width", 6.0f);
  immUniform1f("dash_factor", 0.5f);

  float box_color[4];
  UI_GetThemeColor4fv(TH_VERTEX_SELECT, box_color);
  immUniformColor4fv(box_color);
  imm_draw_box_wire_2d(
      pos, tgpa->rect_cage.xmin, tgpa->rect_cage.ymin, tgpa->rect_cage.xmax, tgpa->rect_cage.ymax);
  immUnbindProgram();

  /* Draw Points. */
  GPU_program_point_size(true);

  immBindBuiltinProgram(GPU_SHADER_2D_POINT_UNIFORM_SIZE_UNIFORM_COLOR_AA);

  float point_color[4];
  UI_GetThemeColor4fv(TH_SELECT, point_color);
  immUniformColor4fv(point_color);

  immUniform1f("size", UI_GetThemeValuef(TH_VERTEX_SIZE) * 1.5f * U.dpi_fac);
  immBegin(GPU_PRIM_POINTS, 8);
  for (int i = 0; i < 8; i++) {
    immVertex2fv(pos, tgpa->manipulator[i]);
  }
  immEnd();
  immUnbindProgram();
  GPU_program_point_size(false);

  GPU_blend(GPU_BLEND_NONE);
}

// TODO
#if 0
static void draw_mouse_position(tGPDfill *tgpf)
{
  if (tgpf->gps_mouse == NULL) {
    return;
  }
  uchar mouse_color[4] = {0, 0, 255, 255};

  bGPDspoint *pt = &tgpf->gps_mouse->points[0];
  float point_size = (tgpf->zoom == 1.0f) ? 4.0f * tgpf->fill_factor :
                                            (0.5f * tgpf->zoom) + tgpf->fill_factor;
  GPUVertFormat *format = immVertexFormat();
  uint pos = GPU_vertformat_attr_add(format, "pos", GPU_COMP_F32, 3, GPU_FETCH_FLOAT);
  uint col = GPU_vertformat_attr_add(format, "color", GPU_COMP_U8, 4, GPU_FETCH_INT_TO_FLOAT_UNIT);

  /* Draw mouse click position in Blue. */
  immBindBuiltinProgram(GPU_SHADER_3D_POINT_FIXED_SIZE_VARYING_COLOR);
  GPU_point_size(point_size);
  immBegin(GPU_PRIM_POINTS, 1);
  immAttr4ubv(col, mouse_color);
  immVertex3fv(pos, &pt->x);
  immEnd();
  immUnbindProgram();
}
#endif

/* Drawing callback for modal operator. */
static void gpencil_asset_draw(const bContext *C, ARegion *UNUSED(region), void *arg)
{
  tGPDasset *tgpa = (tGPDasset *)arg;
  /* Draw only in the region that originated operator. This is required for multi-window. */
  ARegion *region = CTX_wm_region(C);
  if (region != tgpa->region) {
    return;
  }
  gpencil_draw_cage(tgpa);
}
/* ----------------------- */

/* Invoke handler: Initialize the operator */
static int gpencil_asset_import_invoke(bContext *C, wmOperator *op, const wmEvent *event)
{
  wmWindow *win = CTX_wm_window(C);
  bGPdata *gpd = CTX_data_gpencil_data(C);
  tGPDasset *tgpa = NULL;

  /* try to initialize context data needed */
  if (!gpencil_asset_import_init(C, op)) {
    if (op->customdata) {
      MEM_freeN(op->customdata);
    }
    return OPERATOR_CANCELLED;
  }
  tgpa = op->customdata;

  /* Save initial position of drop.  */
  tgpa->drop_x = event->x;
  tgpa->drop_y = event->y;

  /* Do an initial load of the strokes in the target datablock. */
  gpencil_asset_add_strokes(tgpa);

  tgpa->draw_handle_3d = ED_region_draw_cb_activate(
      tgpa->region->type, gpencil_asset_draw, tgpa, REGION_DRAW_POST_PIXEL);

  /* set cursor to indicate modal */
  WM_cursor_modal_set(win, WM_CURSOR_EW_SCROLL);

  /* update shift indicator in header */
  gpencil_asset_import_status_indicators(C, tgpa);
  DEG_id_tag_update(&gpd->id, ID_RECALC_TRANSFORM | ID_RECALC_GEOMETRY);
  WM_event_add_notifier(C, NC_GPENCIL | NA_EDITED, NULL);

  /* add a modal handler for this operator */
  WM_event_add_modal_handler(C, op);

  return OPERATOR_RUNNING_MODAL;
}

/* Modal handler: Events handling during interactive part */
static int gpencil_asset_import_modal(bContext *C, wmOperator *op, const wmEvent *event)
{
  tGPDasset *tgpa = op->customdata;
  wmWindow *win = CTX_wm_window(C);

  switch (event->type) {
    case LEFTMOUSE: {
      if (event->shift) {
        tgpa->mode = GP_ASSET_TRANSFORM_ROT;
      }
      else {
        tgpa->mode = GP_ASSET_TRANSFORM_LOC;
      }
      copy_v2_v2_int(tgpa->mouse, event->mval);
      break;
    }
      /* Confirm */
    case EVT_PADENTER:
    case EVT_RETKEY: {
      /* return to normal cursor and header status */
      ED_area_status_text(tgpa->area, NULL);
      ED_workspace_status_text(C, NULL);
      WM_cursor_modal_restore(win);

      /* Clean up temp data. */
      gpencil_asset_import_exit(C, op);

      /* done! */
      return OPERATOR_FINISHED;
    }

    case EVT_ESCKEY: /* cancel */
    case RIGHTMOUSE: {
      /* return to normal cursor and header status */
      ED_area_status_text(tgpa->area, NULL);
      ED_workspace_status_text(C, NULL);
      WM_cursor_modal_restore(win);

      /* Clean up temp data. */
      gpencil_asset_import_exit(C, op);

      /* canceled! */
      return OPERATOR_CANCELLED;
    }

    case MOUSEMOVE: /* calculate new position */
    {
      /* Update shift based on position of mouse. */
      // TODO

      /* Update screen. */
      gpencil_asset_import_update(C, op, tgpa);
      break;
    }
    case WHEELUPMOUSE: {
      // Scale
      // TODO

      /* Update screen. */
      gpencil_asset_import_update(C, op, tgpa);
      break;
    }
    case WHEELDOWNMOUSE: {
      // Scale
      // TODO

      /* Update screen. */
      gpencil_asset_import_update(C, op, tgpa);
      break;
    }

    default: {
      /* Unhandled event - allow to pass through. */
      return OPERATOR_RUNNING_MODAL | OPERATOR_PASS_THROUGH;
    }
  }
  /* still running... */
  return OPERATOR_RUNNING_MODAL;
}

/* Cancel handler */
static void gpencil_asset_import_cancel(bContext *C, wmOperator *op)
{
  /* this is just a wrapper around exit() */
  gpencil_asset_import_exit(C, op);
}

void GPENCIL_OT_asset_import(wmOperatorType *ot)
{

  PropertyRNA *prop;

  /* identifiers */
  ot->name = "Grease Pencil Import Asset";
  ot->idname = "GPENCIL_OT_asset_import";
  ot->description = "Import Asset into existing grease pencil object";

  /* callbacks */
  ot->invoke = gpencil_asset_import_invoke;
  ot->modal = gpencil_asset_import_modal;
  ot->cancel = gpencil_asset_import_cancel;
  ot->poll = gpencil_asset_generic_poll;

  /* flags */
  ot->flag = OPTYPE_UNDO | OPTYPE_BLOCKING;

  /* Properties. */
  RNA_def_string(ot->srna, "name", "Name", MAX_ID_NAME - 2, "Name", "ID name to add");
  prop = RNA_def_enum(ot->srna, "type", rna_enum_id_type_items, 0, "Type", "");
  RNA_def_property_translation_context(prop, BLT_I18NCONTEXT_ID_ID);
}
/** \} */
