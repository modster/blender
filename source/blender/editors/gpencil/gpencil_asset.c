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
#include "BLI_utildefines.h"

#include "DNA_gpencil_types.h"

#include "BKE_context.h"
#include "BKE_gpencil.h"
#include "BKE_lib_id.h"
#include "BKE_main.h"
#include "BKE_report.h"

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
