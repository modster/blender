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
#include "DNA_material_types.h"

#include "BKE_context.h"
#include "BKE_gpencil.h"
#include "BKE_gpencil_geom.h"
#include "BKE_lib_id.h"
#include "BKE_main.h"
#include "BKE_material.h"
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

#define ROTATION_CONTROL_GAP 15.0f

/* Temporary Asset import operation data */
typedef struct tGPDasset {
  struct wmWindow *win;
  struct Main *bmain;
  struct Depsgraph *depsgraph;
  struct Scene *scene;
  struct ScrArea *area;
  struct ARegion *region;
  struct RegionView3D *rv3d;
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
  int drop[2];
  /** Mouse last click position. */
  int mouse[2];
  /** Initial distance to asset center from mouse location. */
  float initial_dist;
  /** Asset center. */
  float asset_center[3];

  /** 2D Cage vertices. */
  rctf rect_cage;
  /** 2D cage center. */
  float cage_center[2];
  /** 2D cage manipulator points *
   *
   *               8 (Rotation)
   *              /
   *   0----1----2
   *   |         |
   *   7         3
   *   |         |
   *   6----5----4
   */
  float manipulator[9][2];
  /** Manipulator index (-1 means not set). */
  int manipulator_index;
  /** Manipulator vector used to determine the effect. */
  float manipulator_vector[3];
  /** Normal vector for cage. */
  float normal_vec[3];
  /** Vector with the original orientation for rotation. */
  float vinit_rotation[2];

  /** Hash of new created layers. */
  struct GHash *asset_layers;
  /** Hash of new created frames. */
  struct GHash *asset_frames;
  /** Hash of new created strokes. */
  struct GHash *asset_strokes;
  /** Hash of new created strokes linked to layer. */
  struct GHash *asset_strokes_layer;

  /** Handle for drawing while operator is running. */
  void *draw_handle_3d;

} tGPDasset;

typedef enum eGP_AssetFlag {
  /* Waiting for doing something. */
  GP_ASSET_FLAG_IDLE = (1 << 0),
  /* Doing a transform. */
  GP_ASSET_FLAG_TRANSFORMING = (1 << 1),
} eGP_AssetFlag;

typedef enum eGP_AssetTransformMode {
  /* NO action. */
  GP_ASSET_TRANSFORM_NONE = 0,
  /* Location. */
  GP_ASSET_TRANSFORM_LOC = 1,
  /* Rotation. */
  GP_ASSET_TRANSFORM_ROT = 2,
  /* Scale. */
  GP_ASSET_TRANSFORM_SCALE = 3,
} eGP_AssetTransformMode;

enum eGP_CageCorners {
  CAGE_CORNER_NW = 0,
  CAGE_CORNER_N = 1,
  CAGE_CORNER_NE = 2,
  CAGE_CORNER_E = 3,
  CAGE_CORNER_SE = 4,
  CAGE_CORNER_S = 5,
  CAGE_CORNER_SW = 6,
  CAGE_CORNER_W = 7,
  CAGE_CORNER_ROT = 8,
};

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

  const eGP_AssetModes mode = RNA_enum_get(op->ptr, "mode");
  const int reset_origin = RNA_boolean_get(op->ptr, "reset_origin");

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
    /* Remove parenting data. */
    gpl->parent = NULL;
    gpl->parsubstr[0] = 0;
    gpl->partype = 0;

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

  /* Set origin to bounding box of  strokes. */
  if (reset_origin) {
    float gpcenter[3];
    BKE_gpencil_centroid_3d(gpd, gpcenter);

    LISTBASE_FOREACH (bGPDlayer *, gpl, &gpd->layers) {
      LISTBASE_FOREACH (bGPDframe *, gpf, &gpl->frames) {
        LISTBASE_FOREACH (bGPDstroke *, gps, &gpf->strokes) {
          bGPDspoint *pt;
          int i;
          for (i = 0, pt = gps->points; i < gps->totpoints; i++, pt++) {
            sub_v3_v3(&pt->x, gpcenter);
          }
          BKE_gpencil_stroke_boundingbox_calc(gps);
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
  RNA_def_boolean(ot->srna,
                  "reset_origin",
                  1,
                  "Reset Origin to Strokes",
                  "Set origin of the strokes in the center of the bounding box");
}

/** \} */
/* -------------------------------------------------------------------- */
/** \name Import Grease Pencil Asset into existing datablock operator
 * \{ */

/* Helper: Update all imported strokes */
static void gpencil_asset_import_update_strokes(bContext *C, tGPDasset *tgpa)
{
  bGPdata *gpd = tgpa->gpd;

  DEG_id_tag_update(&gpd->id, ID_RECALC_TRANSFORM | ID_RECALC_GEOMETRY);
  WM_event_add_notifier(C, NC_GPENCIL | NA_EDITED, NULL);
}

/* Helper: Draw status message while the user is running the operator */
static void gpencil_asset_import_status_indicators(bContext *C, tGPDasset *tgpa)
{
  char status_str[UI_MAX_DRAW_STR];
  char msg_str[UI_MAX_DRAW_STR];
  bGPdata *gpd_asset = tgpa->gpd_asset;
  const char *mode_txt[] = {"", "(Location)", "(Rotation)", "(Scale)"};

  BLI_strncpy(msg_str, TIP_("Importing Asset"), UI_MAX_DRAW_STR);

  BLI_snprintf(status_str,
               sizeof(status_str),
               "%s %s %s",
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
  /* Update shift indicator in header. */
  gpencil_asset_import_status_indicators(C, tgpa);
  /* Update points position. */
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
      BLI_ghash_free(tgpa->asset_strokes_layer, NULL, NULL);
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
  tgpa->win = CTX_wm_window(C);
  tgpa->bmain = CTX_data_main(C);
  tgpa->depsgraph = CTX_data_ensure_evaluated_depsgraph(C);
  tgpa->scene = CTX_data_scene(C);
  tgpa->area = CTX_wm_area(C);
  tgpa->region = CTX_wm_region(C);
  tgpa->rv3d = CTX_wm_region_view3d(C);
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
  tgpa->flag |= GP_ASSET_FLAG_IDLE;

  /* Manipulator point is not set yet. */
  tgpa->manipulator_index = -1;

  tgpa->asset_layers = NULL;
  tgpa->asset_frames = NULL;
  tgpa->asset_strokes = NULL;
  tgpa->asset_strokes_layer = NULL;

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
static void gpencil_2d_cage_calc(tGPDasset *tgpa)
{
  /* Add some oversize. */
  const float oversize = 5.0f;

  float diff_mat[4][4];
  float cage_min[2];
  float cage_max[2];
  INIT_MINMAX2(cage_min, cage_max);

  GHashIterator gh_iter;
  GHASH_ITER (gh_iter, tgpa->asset_strokes_layer) {
    // TODO: All strokes or only active frame?
    bGPDstroke *gps = (bGPDstroke *)BLI_ghashIterator_getKey(&gh_iter);
    bGPDlayer *gpl = (bGPDlayer *)BLI_ghashIterator_getValue(&gh_iter);
    BKE_gpencil_layer_transform_matrix_get(tgpa->depsgraph, tgpa->ob, gpl, diff_mat);

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

  /* Cage center. */
  tgpa->cage_center[0] = 0.5f * (tgpa->rect_cage.xmin + tgpa->rect_cage.xmax);
  tgpa->cage_center[1] = 0.5f * (tgpa->rect_cage.ymin + tgpa->rect_cage.ymax);

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
  /* Rotation */
  tgpa->manipulator[8][0] = tgpa->rect_cage.xmax + ROTATION_CONTROL_GAP;
  tgpa->manipulator[8][1] = tgpa->rect_cage.ymax + ROTATION_CONTROL_GAP;

  /* Normal vector. */
  float co1[3], co2[3], co3[3], vec1[3], vec2[3];
  gpencil_point_xy_to_3d(&tgpa->gsc, tgpa->scene, tgpa->manipulator[CAGE_CORNER_NE], co1);
  gpencil_point_xy_to_3d(&tgpa->gsc, tgpa->scene, tgpa->manipulator[CAGE_CORNER_NW], co2);
  gpencil_point_xy_to_3d(&tgpa->gsc, tgpa->scene, tgpa->manipulator[CAGE_CORNER_SW], co3);
  sub_v3_v3v3(vec1, co2, co1);
  sub_v3_v3v3(vec2, co3, co2);
  /* Vector orthogonal to polygon plane. */
  cross_v3_v3v3(tgpa->normal_vec, vec1, vec2);
  normalize_v3(tgpa->normal_vec);
}

/* Helper: Detect mouse over cage areas. */
static void gpencil_2d_cage_area_detect(tGPDasset *tgpa, const int mouse[2])
{
  const float gap = 5.0f;

  /* Check if over any of the corners for scale with a small gap. */
  rctf rect_mouse = {
      (float)mouse[0] - gap, (float)mouse[0] + gap, (float)mouse[1] - gap, (float)mouse[1] + gap};

  tgpa->manipulator_index = -1;
  float co1[3], co2[3];
  for (int i = 0; i < 9; i++) {
    if (BLI_rctf_isect_pt(&rect_mouse, tgpa->manipulator[i][0], tgpa->manipulator[i][1])) {
      tgpa->manipulator_index = i;
      tgpa->mode = (tgpa->manipulator_index < 8) ? GP_ASSET_TRANSFORM_SCALE :
                                                   GP_ASSET_TRANSFORM_ROT;

      if (tgpa->mode == GP_ASSET_TRANSFORM_ROT) {
        WM_cursor_modal_set(tgpa->win, WM_CURSOR_HAND);
        return;
      }

      switch (i) {
        case CAGE_CORNER_NW:
        case CAGE_CORNER_SW:
        case CAGE_CORNER_NE:
        case CAGE_CORNER_SE:
          WM_cursor_modal_set(tgpa->win, WM_CURSOR_NSEW_SCROLL);
          break;
        case CAGE_CORNER_N:
        case CAGE_CORNER_S:
          WM_cursor_modal_set(tgpa->win, WM_CURSOR_NS_ARROW);
          break;
        case CAGE_CORNER_E:
        case CAGE_CORNER_W:
          WM_cursor_modal_set(tgpa->win, WM_CURSOR_EW_ARROW);
          break;
        default:
          break;
      }
      /* Determine the vector of the cage effect. For corners is always full effect. */
      if (ELEM(tgpa->manipulator_index,
               CAGE_CORNER_NW,
               CAGE_CORNER_NE,
               CAGE_CORNER_SE,
               CAGE_CORNER_SW)) {
        zero_v3(tgpa->manipulator_vector);
        add_v3_fl(tgpa->manipulator_vector, 1.0f);
        return;
      }
      if (ELEM(tgpa->manipulator_index, CAGE_CORNER_N, CAGE_CORNER_S)) {
        gpencil_point_xy_to_3d(&tgpa->gsc, tgpa->scene, tgpa->manipulator[CAGE_CORNER_S], co1);
        gpencil_point_xy_to_3d(&tgpa->gsc, tgpa->scene, tgpa->manipulator[CAGE_CORNER_N], co2);
      }
      else if (ELEM(tgpa->manipulator_index, CAGE_CORNER_E, CAGE_CORNER_W)) {
        gpencil_point_xy_to_3d(&tgpa->gsc, tgpa->scene, tgpa->manipulator[CAGE_CORNER_W], co1);
        gpencil_point_xy_to_3d(&tgpa->gsc, tgpa->scene, tgpa->manipulator[CAGE_CORNER_E], co2);
      }

      sub_v3_v3v3(tgpa->manipulator_vector, co2, co1);
      normalize_v3(tgpa->manipulator_vector);

      return;
    }
  }

  /* Check if mouse is inside cage for Location. */
  if (BLI_rctf_isect_pt(&tgpa->rect_cage, (float)mouse[0], (float)mouse[1])) {
    tgpa->mode = GP_ASSET_TRANSFORM_LOC;
    WM_cursor_modal_set(tgpa->win, WM_CURSOR_DEFAULT);
    return;
  }

  tgpa->mode = GP_ASSET_TRANSFORM_NONE;
  WM_cursor_modal_set(tgpa->win, WM_CURSOR_DEFAULT);
}

/* Helper: Get the rotation matrix for the angle using an arbitrary vector as axis. */
static void gpencil_asset_rotation_matrix_get(float angle,
                                              float axis[3],
                                              float rotation_matrix[4][4])
{
  float u2 = axis[0] * axis[0];
  float v2 = axis[1] * axis[1];
  float w2 = axis[2] * axis[2];
  const float length = (u2 + v2 + w2);
  const float length_sqr = sqrt(length);
  const float cos_value = cos(angle);
  const float sin_value = sin(angle);

  rotation_matrix[0][0] = (u2 + (v2 + w2) * cos_value) / length;
  rotation_matrix[0][1] = (axis[0] * axis[1] * (1.0f - cos_value) -
                           axis[2] * length_sqr * sin_value) /
                          length;
  rotation_matrix[0][2] = (axis[0] * axis[2] * (1.0f - cos_value) +
                           axis[1] * length_sqr * sin_value) /
                          length;
  rotation_matrix[0][3] = 0.0;

  rotation_matrix[1][0] = (axis[0] * axis[1] * (1.0f - cos_value) +
                           axis[2] * length_sqr * sin_value) /
                          length;
  rotation_matrix[1][1] = (v2 + (u2 + w2) * cos_value) / length;
  rotation_matrix[1][2] = (axis[1] * axis[2] * (1.0f - cos_value) -
                           axis[0] * length_sqr * sin_value) /
                          length;
  rotation_matrix[1][3] = 0.0f;

  rotation_matrix[2][0] = (axis[0] * axis[2] * (1.0f - cos_value) -
                           axis[1] * length_sqr * sin_value) /
                          length;
  rotation_matrix[2][1] = (axis[1] * axis[2] * (1.0f - cos_value) +
                           axis[0] * length_sqr * sin_value) /
                          length;
  rotation_matrix[2][2] = (w2 + (u2 + v2) * cos_value) / length;
  rotation_matrix[2][3] = 0.0f;

  rotation_matrix[3][0] = 0.0f;
  rotation_matrix[3][1] = 0.0f;
  rotation_matrix[3][2] = 0.0f;
  rotation_matrix[3][3] = 1.0f;
}

/* Helper: Transfrom the stroke with mouse movements. */
static void gpencil_asset_transform_strokes(tGPDasset *tgpa,
                                            const int mouse[2],
                                            const bool shift_key)
{
  /* Get the vector with the movement done by the mouse since last event. */
  float origin_pt[3], dest_pt[3];
  float mousef[2];
  copy_v2fl_v2i(mousef, tgpa->mouse);
  gpencil_point_xy_to_3d(&tgpa->gsc, tgpa->scene, mousef, origin_pt);

  copy_v2fl_v2i(mousef, mouse);
  gpencil_point_xy_to_3d(&tgpa->gsc, tgpa->scene, mousef, dest_pt);

  float vec[3];
  sub_v3_v3v3(vec, dest_pt, origin_pt);

  /* Get the scale factor. */
  float mouse3d[3];
  sub_v3_v3v3(mouse3d, dest_pt, tgpa->asset_center);
  float dist = len_v3(mouse3d);
  float scale_factor = dist / tgpa->initial_dist;
  float scale_vector[3];
  mul_v3_v3fl(scale_vector, tgpa->manipulator_vector, scale_factor - 1.0f);
  add_v3_fl(scale_vector, 1.0f);

  /* Determine pivot point. */
  float pivot[3];
  copy_v3_v3(pivot, tgpa->asset_center);
  if (!shift_key) {
    if (tgpa->manipulator_index == CAGE_CORNER_N) {
      gpencil_point_xy_to_3d(&tgpa->gsc, tgpa->scene, tgpa->manipulator[CAGE_CORNER_S], pivot);
    }
    else if (tgpa->manipulator_index == CAGE_CORNER_E) {
      gpencil_point_xy_to_3d(&tgpa->gsc, tgpa->scene, tgpa->manipulator[CAGE_CORNER_W], pivot);
    }
    else if (tgpa->manipulator_index == CAGE_CORNER_S) {
      gpencil_point_xy_to_3d(&tgpa->gsc, tgpa->scene, tgpa->manipulator[CAGE_CORNER_N], pivot);
    }
    else if (tgpa->manipulator_index == CAGE_CORNER_W) {
      gpencil_point_xy_to_3d(&tgpa->gsc, tgpa->scene, tgpa->manipulator[CAGE_CORNER_E], pivot);
    }
  }

  /* Create rotation matrix. */
  float rot_matrix[4][4];
  float vr[2];
  copy_v2fl_v2i(vr, mouse);
  sub_v2_v2v2(vr, vr, tgpa->cage_center);
  normalize_v2(vr);
  float angle = angle_signed_v2v2(tgpa->vinit_rotation, vr);
  gpencil_asset_rotation_matrix_get(angle, tgpa->normal_vec, rot_matrix);

  GHashIterator gh_iter;
  GHASH_ITER (gh_iter, tgpa->asset_strokes) {
    bGPDstroke *gps = (bGPDstroke *)BLI_ghashIterator_getKey(&gh_iter);
    bGPDspoint *pt;
    int i;
    for (i = 0, pt = gps->points; i < gps->totpoints; i++, pt++) {
      switch (tgpa->mode) {
        case GP_ASSET_TRANSFORM_LOC: {
          add_v3_v3(&pt->x, vec);
          break;
        }
        case GP_ASSET_TRANSFORM_ROT: {
          sub_v3_v3(&pt->x, pivot);
          mul_v3_m4v3(&pt->x, rot_matrix, &pt->x);
          add_v3_v3(&pt->x, pivot);
          break;
        }
        case GP_ASSET_TRANSFORM_SCALE: {
          /* Apply scale. */
          sub_v3_v3(&pt->x, pivot);
          mul_v3_v3(&pt->x, scale_vector);
          add_v3_v3(&pt->x, pivot);

          /* Thickness change only in full scale. */
          if (ELEM(tgpa->manipulator_index,
                   CAGE_CORNER_NW,
                   CAGE_CORNER_NE,
                   CAGE_CORNER_SE,
                   CAGE_CORNER_SW)) {
            pt->pressure *= scale_factor;
            CLAMP_MIN(pt->pressure, 0.01f);
          }
          break;
        }
        default:
          break;
      }
    }

    /* In scale mode recal geometry. */
    if (tgpa->mode == GP_ASSET_TRANSFORM_SCALE) {
      BKE_gpencil_stroke_geometry_update(tgpa->gpd, gps);
    }
    else {
      /* Recalc stroke bounding box. */
      BKE_gpencil_stroke_boundingbox_calc(gps);
    }
  }

  /* In location mode move the asset center. */
  if (tgpa->mode == GP_ASSET_TRANSFORM_LOC) {
    add_v3_v3(tgpa->asset_center, vec);
  }

  /* Update mouse position and distance to calc the factor to transform. */
  copy_v2_v2_int(tgpa->mouse, mouse);
  tgpa->initial_dist = dist;
}

static Material *gpencil_asset_material_get_from_id(ID *id, const int slot_index)
{
  const short *tot_slots_data_ptr = BKE_id_material_len_p(id);
  const int tot_slots_data = tot_slots_data_ptr ? *tot_slots_data_ptr : 0;
  if (slot_index >= tot_slots_data) {
    return NULL;
  }

  Material ***materials_data_ptr = BKE_id_material_array_p(id);
  Material **materials_data = materials_data_ptr ? *materials_data_ptr : NULL;
  Material *material = materials_data[slot_index];

  return material;
}

/* Helper: Load all strokes in the target datablock. */
static void gpencil_asset_add_strokes(tGPDasset *tgpa)
{
  bGPdata *gpd_target = tgpa->gpd;
  bGPdata *gpd_asset = tgpa->gpd_asset;

  /* Get the vector from origin to drop position. */
  float dest_pt[3];
  float loc2d[2];
  copy_v2fl_v2i(loc2d, tgpa->drop);
  gpencil_point_xy_to_3d(&tgpa->gsc, tgpa->scene, loc2d, dest_pt);

  float vec[3];
  sub_v3_v3v3(vec, dest_pt, tgpa->ob->loc);

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
        tgpa->asset_strokes_layer = BLI_ghash_ptr_new(__func__);
      }

      LISTBASE_FOREACH (bGPDstroke *, gps_asset, &gpf_asset->strokes) {
        bGPDstroke *gps_target = BKE_gpencil_stroke_duplicate(gps_asset, true, true);
        gps_target->flag &= ~GP_STROKE_SELECT;
        BLI_addtail(&gpf_target->strokes, gps_target);

        /* Add the material. */
        Material *ma_src = gpencil_asset_material_get_from_id(&tgpa->gpd_asset->id,
                                                              gps_asset->mat_nr);

        int mat_index = BKE_gpencil_object_material_index_get_by_name(tgpa->ob,
                                                                      ma_src->id.name + 2);
        if (mat_index == -1) {
          mat_index = BKE_gpencil_object_material_ensure(tgpa->bmain, tgpa->ob, ma_src);
        }

        gps_target->mat_nr = mat_index;

        /* Apply the offset to drop position. */
        bGPDspoint *pt;
        int i;
        for (i = 0, pt = gps_target->points; i < gps_target->totpoints; i++, pt++) {
          add_v3_v3(&pt->x, vec);
          pt->flag &= ~GP_SPOINT_SELECT;
        }

        /* Update geometry. */
        BKE_gpencil_stroke_geometry_update(gpd_target, gps_target);

        /* Add the hash key with a reference to the frame. */
        BLI_ghash_insert(tgpa->asset_strokes, gps_target, gpf_target);
        BLI_ghash_insert(tgpa->asset_strokes_layer, gps_target, gpl_target);
      }
    }
  }
  /* Prepare 2D cage. */
  gpencil_2d_cage_calc(tgpa);
  BKE_gpencil_centroid_3d(tgpa->gpd_asset, tgpa->asset_center);
  add_v3_v3(tgpa->asset_center, vec);
}

/* Helper: Clean any temp data. */
static void gpencil_asset_clean_data(tGPDasset *tgpa)
{
  GHashIterator gh_iter;
  /* Clean Strokes. */
  if (tgpa->asset_strokes != NULL) {
    GHASH_ITER (gh_iter, tgpa->asset_strokes) {
      bGPDstroke *gps = (bGPDstroke *)BLI_ghashIterator_getKey(&gh_iter);
      bGPDframe *gpf = (bGPDframe *)BLI_ghashIterator_getValue(&gh_iter);
      BLI_remlink(&gpf->strokes, gps);
      BKE_gpencil_free_stroke(gps);
    }
  }
  /* Clean Frames. */
  if (tgpa->asset_frames != NULL) {
    GHASH_ITER (gh_iter, tgpa->asset_frames) {
      bGPDframe *gpf = (bGPDframe *)BLI_ghashIterator_getKey(&gh_iter);
      bGPDlayer *gpl = (bGPDlayer *)BLI_ghashIterator_getValue(&gh_iter);
      BLI_remlink(&gpl->frames, gpf);
    }
  }
  /* Clean Layers. */
  if (tgpa->asset_layers != NULL) {
    GHASH_ITER (gh_iter, tgpa->asset_layers) {
      bGPDlayer *gpl = (bGPDlayer *)BLI_ghashIterator_getKey(&gh_iter);
      BKE_gpencil_layer_delete(tgpa->gpd, gpl);
    }
  }
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

  immBegin(GPU_PRIM_LINE_LOOP, 4);
  immVertex2f(pos, tgpa->manipulator[CAGE_CORNER_NW][0], tgpa->manipulator[CAGE_CORNER_NW][1]);
  immVertex2f(pos, tgpa->manipulator[CAGE_CORNER_NE][0], tgpa->manipulator[CAGE_CORNER_NE][1]);
  immVertex2f(pos, tgpa->manipulator[CAGE_CORNER_SE][0], tgpa->manipulator[CAGE_CORNER_SE][1]);
  immVertex2f(pos, tgpa->manipulator[CAGE_CORNER_SW][0], tgpa->manipulator[CAGE_CORNER_SW][1]);
  immEnd();

  /* Rotation box */
  /*const float gap = 5.0f;
  imm_draw_box_wire_2d(pos,
                       tgpa->manipulator[CAGE_CORNER_ROT][0] - gap,
                       tgpa->manipulator[CAGE_CORNER_ROT][1] - gap,
                       tgpa->manipulator[CAGE_CORNER_ROT][0] + gap,
                       tgpa->manipulator[CAGE_CORNER_ROT][1] + gap);*/

  immBegin(GPU_PRIM_LINES, 2);
  immVertex2f(pos, tgpa->manipulator[CAGE_CORNER_NE][0], tgpa->manipulator[CAGE_CORNER_NE][1]);
  immVertex2f(pos, tgpa->manipulator[CAGE_CORNER_ROT][0], tgpa->manipulator[CAGE_CORNER_ROT][1]);
  immEnd();

  immUnbindProgram();

  /* Draw Points. */
  GPU_program_point_size(true);

  immBindBuiltinProgram(GPU_SHADER_2D_POINT_UNIFORM_SIZE_UNIFORM_COLOR_AA);

  float point_color[4];
  UI_GetThemeColor4fv(TH_SELECT, point_color);
  immUniformColor4fv(point_color);

  immUniform1f("size", UI_GetThemeValuef(TH_VERTEX_SIZE) * 1.5f * U.dpi_fac);
  /* Draw points. */
  immBegin(GPU_PRIM_POINTS, 9);
  for (int i = 0; i < 9; i++) {
    immVertex2fv(pos, tgpa->manipulator[i]);
  }
  immEnd();
  immUnbindProgram();
  GPU_program_point_size(false);

  GPU_blend(GPU_BLEND_NONE);
}

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
  tgpa->drop[0] = event->mval[0];
  tgpa->drop[1] = event->mval[1];

  /* Do an initial load of the strokes in the target datablock. */
  gpencil_asset_add_strokes(tgpa);

  tgpa->draw_handle_3d = ED_region_draw_cb_activate(
      tgpa->region->type, gpencil_asset_draw, tgpa, REGION_DRAW_POST_PIXEL);

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
      /* If click ouside cage, confirm. */
      if (event->val == KM_PRESS) {
        rctf rect_big;
        rect_big.xmin = tgpa->rect_cage.xmin - (ROTATION_CONTROL_GAP * 2.0f);
        rect_big.ymin = tgpa->rect_cage.ymin - (ROTATION_CONTROL_GAP * 2.0f);
        rect_big.xmax = tgpa->rect_cage.xmax + (ROTATION_CONTROL_GAP * 2.0f);
        rect_big.ymax = tgpa->rect_cage.ymax + (ROTATION_CONTROL_GAP * 2.0f);

        if ((tgpa->flag & GP_ASSET_FLAG_IDLE) &&
            (!BLI_rctf_isect_pt(&rect_big, (float)event->mval[0], (float)event->mval[1]))) {
          ED_area_status_text(tgpa->area, NULL);
          ED_workspace_status_text(C, NULL);
          WM_cursor_modal_restore(win);
          gpencil_asset_import_exit(C, op);
          return OPERATOR_FINISHED;
        }
      }

      if (event->val == KM_RELEASE) {
        tgpa->flag |= GP_ASSET_FLAG_IDLE;
        tgpa->flag &= ~GP_ASSET_FLAG_TRANSFORMING;
        tgpa->mode = GP_ASSET_TRANSFORM_NONE;
        WM_cursor_modal_set(tgpa->win, WM_CURSOR_DEFAULT);
        break;
      }

      if (tgpa->flag & GP_ASSET_FLAG_IDLE) {
        copy_v2_v2_int(tgpa->mouse, event->mval);

        /* Distance to asset center. */
        float mousef[2], mouse3d[3];
        copy_v2fl_v2i(mousef, tgpa->mouse);
        gpencil_point_xy_to_3d(&tgpa->gsc, tgpa->scene, mousef, mouse3d);
        sub_v3_v3v3(mouse3d, mouse3d, tgpa->asset_center);
        tgpa->initial_dist = len_v3(mouse3d);

        /* Initial orientation for rotation. */
        copy_v2fl_v2i(tgpa->vinit_rotation, tgpa->mouse);
        sub_v2_v2v2(tgpa->vinit_rotation, tgpa->vinit_rotation, tgpa->cage_center);
        normalize_v2(tgpa->vinit_rotation);

        tgpa->flag &= ~GP_ASSET_FLAG_IDLE;
        tgpa->flag |= GP_ASSET_FLAG_TRANSFORMING;
      }
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
      /* Delete temp strokes. */
      gpencil_asset_clean_data(tgpa);
      /* Return to normal cursor and header status */
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
      /* Apply transform. */
      if (tgpa->flag & GP_ASSET_FLAG_TRANSFORMING) {
        gpencil_asset_transform_strokes(tgpa, event->mval, event->shift);
        gpencil_2d_cage_calc(tgpa);
        ED_area_tag_redraw(tgpa->area);
      }
      else {
        /* Check cage manipulators. */
        gpencil_2d_cage_area_detect(tgpa, event->mval);
      }
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
