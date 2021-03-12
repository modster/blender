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
 * The Original Code is Copyright (C) 2008, Blender Foundation
 * This is a new part of Blender
 * Operators for editing Grease Pencil strokes
 */

/** \file
 * \ingroup edgpencil
 */

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "DNA_gpencil_types.h"
#include "DNA_view3d_types.h"

#include "BKE_context.h"
#include "BKE_gpencil.h"
#include "BKE_gpencil_curve.h"
#include "BKE_gpencil_geom.h"
#include "BKE_report.h"

#include "BLI_listbase.h"
#include "BLI_math.h"

#include "RNA_access.h"
#include "RNA_define.h"

#include "WM_api.h"
#include "WM_types.h"

#include "ED_gpencil.h"

#include "DEG_depsgraph.h"

#include "UI_interface.h"
#include "UI_resources.h"

#include "gpencil_intern.h"

/* -------------------------------------------------------------------- */
/** \name Set handle type operator
 * \{ */

static int gpencil_editcurve_set_handle_type_exec(bContext *C, wmOperator *op)
{
  Object *ob = CTX_data_active_object(C);
  bGPdata *gpd = ob->data;
  const int handle_type = RNA_enum_get(op->ptr, "type");

  if (ELEM(NULL, gpd)) {
    return OPERATOR_CANCELLED;
  }

  GP_EDITABLE_CURVES_BEGIN(gps_iter, C, gpl, gps, gpc)
  {
    for (int i = 0; i < gpc->tot_curve_points; i++) {
      bGPDcurve_point *gpc_pt = &gpc->curve_points[i];

      if (gpc_pt->flag & GP_CURVE_POINT_SELECT) {
        BezTriple *bezt = &gpc_pt->bezt;

        if (bezt->f2 & SELECT) {
          bezt->h1 = handle_type;
          bezt->h2 = handle_type;
        }
        else {
          if (bezt->f1 & SELECT) {
            bezt->h1 = handle_type;
          }
          if (bezt->f3 & SELECT) {
            bezt->h2 = handle_type;
          }
        }
      }
    }

    BKE_gpencil_editcurve_recalculate_handles(gps);
    gps->flag |= GP_STROKE_NEEDS_CURVE_UPDATE;
    BKE_gpencil_stroke_geometry_update(gpd, gps);
  }
  GP_EDITABLE_CURVES_END(gps_iter);

  /* notifiers */
  DEG_id_tag_update(&gpd->id, ID_RECALC_TRANSFORM | ID_RECALC_GEOMETRY);
  WM_event_add_notifier(C, NC_GPENCIL | ND_DATA | NA_EDITED, NULL);

  return OPERATOR_FINISHED;
}

void GPENCIL_OT_stroke_editcurve_set_handle_type(wmOperatorType *ot)
{
  static const EnumPropertyItem editcurve_handle_type_items[] = {
      {HD_FREE, "FREE", 0, "Free", ""},
      {HD_AUTO, "AUTOMATIC", 0, "Automatic", ""},
      {HD_VECT, "VECTOR", 0, "Vector", ""},
      {HD_ALIGN, "ALIGNED", 0, "Aligned", ""},
      {0, NULL, 0, NULL, NULL},
  };

  /* identifiers */
  ot->name = "Set handle type";
  ot->idname = "GPENCIL_OT_stroke_editcurve_set_handle_type";
  ot->description = "Set the type of a edit curve handle";

  /* api callbacks */
  ot->invoke = WM_menu_invoke;
  ot->exec = gpencil_editcurve_set_handle_type_exec;
  ot->poll = gpencil_active_layer_poll;

  /* flags */
  ot->flag = OPTYPE_REGISTER | OPTYPE_UNDO;

  /* properties */
  ot->prop = RNA_def_enum(ot->srna, "type", editcurve_handle_type_items, 1, "Type", "Spline type");
}
/** \} */

/* -------------------------------------------------------------------- */
/** \name Set stroke type operator
 * \{ */

static int gpencil_stroke_set_type_exec(bContext *C, wmOperator *op)
{
  Object *ob = CTX_data_active_object(C);
  bGPdata *gpd = ob->data;
  eGPStrokeType type = RNA_enum_get(op->ptr, "type");
  const bool only_selected = RNA_boolean_get(op->ptr, "only_selected");

  if (ELEM(NULL, gpd)) {
    return OPERATOR_CANCELLED;
  }

  bool changed = false;
  switch (type) {
    case STROKE_POLY: {
      GP_EDITABLE_CURVES_BEGIN(gps_iter, C, gpl, gps, gpc)
      {
        if (only_selected && (gpc->flag & GP_CURVE_SELECT) == 0) {
          continue;
        }
        BKE_gpencil_stroke_editcurve_sync_selection(gpd, gps, gpc);
        BKE_gpencil_free_stroke_editcurve(gps);
        changed = true;
      }
      GP_EDITABLE_CURVES_END(gps_iter);
    } break;

    case STROKE_BEZIER: {
      const float threshold = RNA_float_get(op->ptr, "threshold");
      const float corner_angle = RNA_float_get(op->ptr, "corner_angle");

      GP_EDITABLE_STROKES_BEGIN (gps_iter, C, gpl, gps) {
        if (!GPENCIL_STROKE_TYPE_BEZIER(gps)) {
          if (only_selected && (gps->flag & GP_STROKE_SELECT) == 0) {
            continue;
          }

          BKE_gpencil_stroke_editcurve_update(gps, threshold, corner_angle, false);
          if (gps->editcurve != NULL) {
            bGPDcurve *gpc = gps->editcurve;
            gps->flag |= GP_STROKE_NEEDS_CURVE_UPDATE;
            BKE_gpencil_stroke_geometry_update(gpd, gps);

            /* Select all curve points. */
            for (uint32_t i = 0; i < gpc->tot_curve_points; i++) {
              bGPDcurve_point *pt = &gpc->curve_points[i];
              pt->flag |= GP_CURVE_POINT_SELECT;
              BEZT_SEL_ALL(&pt->bezt);
            }
            gpc->flag |= GP_CURVE_SELECT;

            /* Deselect stroke points. */
            for (uint32_t i = 0; i < gps->totpoints; i++) {
              bGPDspoint *pt = &gps->points[i];
              pt->flag &= ~GP_SPOINT_SELECT;
            }
            gps->flag &= ~GP_STROKE_SELECT;
            changed = true;
          }
        }
      }
      GP_EDITABLE_STROKES_END(gps_iter);
    } break;
    default: {
      BKE_report(op->reports, RPT_ERROR, "Unknown stroke type");
      return OPERATOR_CANCELLED;
    }
  }

  if (changed) {
    /* notifiers */
    DEG_id_tag_update(&gpd->id, ID_RECALC_TRANSFORM | ID_RECALC_GEOMETRY);
    WM_event_add_notifier(C, NC_GPENCIL | ND_DATA | NA_EDITED, NULL);
  }

  return OPERATOR_FINISHED;
}

static void gpencil_stroke_set_type_ui(bContext *UNUSED(C), wmOperator *op)
{
  uiLayout *layout = op->layout;
  PointerRNA ptr;
  uiLayoutSetPropSep(layout, true);
  RNA_pointer_create(NULL, op->type->srna, op->properties, &ptr);

  uiItemR(layout, &ptr, "only_selected", 0, NULL, ICON_NONE);
  eGPStrokeType type = RNA_enum_get(&ptr, "type");
  if (type == STROKE_BEZIER) {
    uiItemR(layout, &ptr, "threshold", 0, NULL, ICON_NONE);
    uiItemR(layout, &ptr, "corner_angle", 0, NULL, ICON_NONE);
  }
}

void GPENCIL_OT_stroke_set_type(wmOperatorType *ot)
{
  PropertyRNA *prop;
  static const EnumPropertyItem stroke_types[] = {
      {STROKE_POLY, "POLY", 0, "Poly", ""},
      {STROKE_BEZIER, "BEZIER", 0, "Bezier", ""},
      {0, NULL, 0, NULL, NULL},
  };

  /* identifiers */
  ot->name = "Set Stroke Type";
  ot->idname = "GPENCIL_OT_stroke_set_type";
  ot->description = "Set the type of the stroke";

  /* api callbacks */
  ot->exec = gpencil_stroke_set_type_exec;
  ot->poll = gpencil_active_layer_poll;
  ot->ui = gpencil_stroke_set_type_ui;

  /* flags */
  ot->flag = OPTYPE_REGISTER | OPTYPE_UNDO;

  prop = RNA_def_enum(ot->srna, "type", stroke_types, STROKE_POLY, "Type", "Stroke type");
  RNA_def_property_flag(prop, PROP_SKIP_SAVE);

  prop = RNA_def_boolean(
      ot->srna, "only_selected", true, "Selected Only", "Only set the type for selected strokes");

  prop = RNA_def_float(ot->srna,
                       "threshold",
                       GP_DEFAULT_CURVE_ERROR,
                       0.0f,
                       100.0f,
                       "Threshold",
                       "Bézier curve fitting error threshold",
                       0.0f,
                       3.0f);

  prop = RNA_def_float_distance(ot->srna,
                                "corner_angle",
                                GP_DEFAULT_CURVE_EDIT_CORNER_ANGLE,
                                0.0f,
                                M_PI,
                                "Corner Angle",
                                "Angle threshold to be treated as corners",
                                0.0f,
                                M_PI);
  RNA_def_property_subtype(prop, PROP_ANGLE);
}

/** \} */