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

#include "WM_api.h"
#include "WM_types.h"

#include "RNA_access.h"
#include "RNA_define.h"
#include "RNA_enum_types.h"

#include "ED_asset.h"
#include "ED_gpencil.h"
#include "ED_screen.h"

#include "DEG_depsgraph.h"
#include "DEG_depsgraph_build.h"

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

  /** Current frame number. */
  int cframe;
  /** General Flag. */
  int flag;

  /* Drop initial position. */
  int drop_x, drop_y;

  /* Hash of new created layers. */
  struct GHash *used_layers;
  /* Hash of new created frames. */
  struct GHash *used_frames;
  /* Hash of new created strokes. */
  struct GHash *used_strokes;

} tGPDasset;

static bool gpencil_asset_create_poll(bContext *C)
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
  ot->poll = gpencil_asset_create_poll;

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
static void gpencil_asset_import_status_indicators(bContext *C, tGPDasset *p)
{
  char status_str[UI_MAX_DRAW_STR];
  char msg_str[UI_MAX_DRAW_STR];
  bGPdata *gpd_asset = p->gpd_asset;

  BLI_strncpy(msg_str, TIP_("Importing Grease Pencil Asset"), UI_MAX_DRAW_STR);

  BLI_snprintf(status_str, sizeof(status_str), "%s %s", msg_str, gpd_asset->id.name + 2);

  ED_area_status_text(p->area, status_str);
  ED_workspace_status_text(C, TIP_("ESC/RMB to cancel, Enter/LMB to confirm"));
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
    if (tgpa->used_layers != NULL) {
      BLI_ghash_free(tgpa->used_layers, NULL, NULL);
    }
    if (tgpa->used_frames != NULL) {
      BLI_ghash_free(tgpa->used_frames, NULL, NULL);
    }
    if (tgpa->used_strokes != NULL) {
      BLI_ghash_free(tgpa->used_strokes, NULL, NULL);
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
  /* set current scene and window */
  tgpa->depsgraph = CTX_data_ensure_evaluated_depsgraph(C);
  tgpa->scene = CTX_data_scene(C);
  tgpa->area = CTX_wm_area(C);
  tgpa->region = CTX_wm_region(C);
  tgpa->ob = CTX_data_active_object(C);

  /* set current frame number */
  tgpa->cframe = tgpa->scene->r.cfra;

  /* Target GP datablock. */
  tgpa->gpd = tgpa->ob->data;
  /* Asset GP datablock. */
  tgpa->gpd_asset = (bGPdata *)id;

  /* Create Hash tables. */
  tgpa->used_layers = BLI_ghash_ptr_new(__func__);
  tgpa->used_frames = BLI_ghash_ptr_new(__func__);
  tgpa->used_strokes = BLI_ghash_ptr_new(__func__);

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
    case LEFTMOUSE: /* confirm */
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
    } break;

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

static bool gpencil_asset_import_poll(bContext *C)
{
  /* edit only supported with grease pencil objects */
  Object *ob = CTX_data_active_object(C);
  if ((ob == NULL) || (ob->type != OB_GPENCIL)) {
    return false;
  }

  return true;
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
  ot->poll = gpencil_asset_import_poll;

  /* flags */
  ot->flag = OPTYPE_UNDO | OPTYPE_BLOCKING;

  /* Properties. */
  RNA_def_string(ot->srna, "name", "Name", MAX_ID_NAME - 2, "Name", "ID name to add");
  prop = RNA_def_enum(ot->srna, "type", rna_enum_id_type_items, 0, "Type", "");
  RNA_def_property_translation_context(prop, BLT_I18NCONTEXT_ID_ID);

  prop = RNA_def_boolean(ot->srna, "release_confirm", 0, "Confirm on Release", "");
  RNA_def_property_flag(prop, PROP_HIDDEN | PROP_SKIP_SAVE);
}
/** \} */
