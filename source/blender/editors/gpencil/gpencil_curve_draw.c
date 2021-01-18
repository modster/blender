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
 * The Original Code is Copyright (C) 2017, Blender Foundation
 * This is a new part of Blender
 * Operators for creating new Grease Pencil primitives (boxes, circles, ...)
 */

/** \file
 * \ingroup edgpencil
 */
#include <stdio.h>

#include "MEM_guardedalloc.h"

#include "BLI_math.h"

#include "DNA_gpencil_types.h"
#include "DNA_space_types.h"
#include "DNA_windowmanager_types.h"

#include "BKE_context.h"
#include "BKE_gpencil.h"
#include "BKE_gpencil_geom.h"

#include "WM_api.h"
#include "WM_types.h"

#include "RNA_access.h"
#include "RNA_define.h"
#include "RNA_types.h"

#include "ED_gpencil.h"
#include "ED_screen.h"

#include "DEG_depsgraph.h"
#include "DEG_depsgraph_query.h"

#include "gpencil_intern.h"

/* ------------------------------------------------------------------------- */

typedef struct tGPDcurve_draw {
  bGPdata *gpd;
  bGPDframe *gpf;
  bGPDstroke *gps;
  bGPDcurve *gpc;
  int imval[2];
  short flag;
} tGPDcurve_draw;

typedef enum eGPDcurve_draw_state {
  MOUSE_DOWN = (1 << 0),
} eGPDcurve_draw_state;

void gpencil_curve_draw_init(bContext *C, wmOperator *op, const wmEvent *event)
{
  bGPdata *gpd = CTX_data_gpencil_data(C);

  tGPDcurve_draw *tcd = MEM_callocN(sizeof(tGPDcurve_draw), __func__);
  tcd->gpd = gpd;
  copy_v2_v2_int(tcd->imval, event->mval);

  /* Initialize mouse state */
  tcd->flag |= event->val == KM_PRESS ? MOUSE_DOWN : 0;

  op->customdata = tcd;
}

void gpencil_curve_draw_update(bContext *C, tGPDcurve_draw *tcd)
{
  printf("Update curve draw\n");
  bGPdata *gpd = tgpi->gpd;

  DEG_id_tag_update(&gpd->id, ID_RECALC_COPY_ON_WRITE);
  DEG_id_tag_update(&gpd->id, ID_RECALC_TRANSFORM | ID_RECALC_GEOMETRY);
  WM_event_add_notifier(C, NC_GPENCIL | NA_EDITED, NULL);
}

void gpencil_curve_draw_confirm(bContext *C, wmOperator *op, tGPDcurve_draw *tcd)
{
  printf("Confirm curve draw\n");
}

void gpencil_curve_draw_exit(bContext *C, wmOperator *op)
{
  printf("Exit curve draw\n");
  tGPDcurve_draw *tcd = op->customdata;
  bGPdata *gpd = tcd->gpd;

  MEM_SAFE_FREE(tcd);

  DEG_id_tag_update(&gpd->id, ID_RECALC_TRANSFORM | ID_RECALC_GEOMETRY | ID_RECALC_COPY_ON_WRITE);
  WM_event_add_notifier(C, NC_GPENCIL | NA_EDITED, NULL);

  op->customdata = NULL;
}

/* ------------------------------------------------------------------------- */
/* Operator callbacks */

static int gpencil_curve_draw_invoke(bContext *C, wmOperator *op, const wmEvent *event)
{
  printf("Invoke curve draw\n");
  wmWindow *win = CTX_wm_window(C);

  /* Set cursor to dot. */
  WM_cursor_modal_set(win, WM_CURSOR_DOT);

  gpencil_curve_draw_init(C, op, event);
  tGPDcurve_draw *tcd = op->customdata;

  // if (RNA_boolean_get(op->ptr, "wait_for_input") == false) {
  //   printf("%s\tMouse x: %d y: %d\n",
  //          (tcd->flag & MOUSE_DOWN) ? "DOWN" : "UP",
  //          tcd->imval[0],
  //          tcd->imval[1]);
  // }

  /* Add modal handler. */
  WM_event_add_modal_handler(C, op);
  return OPERATOR_RUNNING_MODAL;
}

static int gpencil_curve_draw_modal(bContext *C, wmOperator *op, const wmEvent *event)
{
  tGPDcurve_draw *tcd = op->customdata;
  wmWindow *win = CTX_wm_window(C);

  copy_v2_v2_int(tcd->imval, event->mval);

  switch (event->type) {
    case LEFTMOUSE: {
      if (event->val == KM_PRESS) {
        printf("Mouse press\n");
        tcd->flag |= MOUSE_DOWN;
      }
      else if (event->val == KM_RELEASE) {
        printf("Mouse release\n");
        tcd->flag &= ~MOUSE_DOWN;
      }
      break;
    }
    case RIGHTMOUSE: {
      ATTR_FALLTHROUGH;
    }
    case EVT_ESCKEY: {
      ED_workspace_status_text(C, NULL);
      WM_cursor_modal_restore(win);

      gpencil_curve_draw_exit(C, op);

      return OPERATOR_CANCELLED;
    }
    case EVT_SPACEKEY:
    case MIDDLEMOUSE:
    case EVT_PADENTER:
    case EVT_RETKEY: {
      ED_workspace_status_text(C, NULL);
      WM_cursor_modal_restore(win);

      /* Create curve */
      gpencil_curve_draw_confirm(C, op, tcd);

      gpencil_curve_draw_exit(C, op);

      return OPERATOR_FINISHED;
    }
    case MOUSEMOVE: {
      printf("%s\tMouse x: %d y: %d\n",
             (tcd->flag & MOUSE_DOWN) ? "DOWN" : "UP",
             tcd->imval[0],
             tcd->imval[1]);
      gpencil_curve_draw_update(C, op)
      break;
    }
    default:
      return OPERATOR_RUNNING_MODAL | OPERATOR_PASS_THROUGH;
  }
  return OPERATOR_RUNNING_MODAL;
}

static void gpencil_curve_draw_cancel(bContext *C, wmOperator *op)
{
  printf("Cancel curve draw\n");
  gpencil_curve_draw_exit(C, op);
}

static bool gpencil_curve_draw_poll(bContext *C)
{
  printf("Poll curve draw\n");
  ScrArea *area = CTX_wm_area(C);
  if (area && area->spacetype != SPACE_VIEW3D) {
    return false;
  }

  bGPdata *gpd = CTX_data_gpencil_data(C);
  if (gpd == NULL) {
    return false;
  }

  if ((gpd->flag & GP_DATA_STROKE_PAINTMODE) == 0) {
    return false;
  }

  bGPDlayer *gpl = BKE_gpencil_layer_active_get(gpd);
  if ((gpl) && (gpl->flag & (GP_LAYER_LOCKED | GP_LAYER_HIDE))) {
    return false;
  }

  return true;
}

void GPENCIL_OT_draw_curve(wmOperatorType *ot)
{
  PropertyRNA *prop;

  /* identifiers */
  ot->name = "Grease Pencil Draw Curve";
  ot->idname = "GPENCIL_OT_draw_curve";
  ot->description = "Draw a bezier curve in the active grease pencil object";

  /* api callbacks */
  ot->invoke = gpencil_curve_draw_invoke;
  ot->modal = gpencil_curve_draw_modal;
  ot->cancel = gpencil_curve_draw_cancel;
  ot->poll = gpencil_curve_draw_poll;

  prop = RNA_def_boolean(ot->srna, "wait_for_input", true, "Wait for Input", "");
  RNA_def_property_flag(prop, PROP_HIDDEN | PROP_SKIP_SAVE);

  /* flags */
  ot->flag = OPTYPE_REGISTER | OPTYPE_UNDO | OPTYPE_BLOCKING;
}
