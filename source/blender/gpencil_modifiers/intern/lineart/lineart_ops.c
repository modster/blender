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
 * The Original Code is Copyright (C) 2019 Blender Foundation.
 * All rights reserved.
 */

/** \file
 * \ingroup editors
 */

#include <stdlib.h>

#include "BKE_collection.h"
#include "BKE_context.h"
#include "BKE_gpencil.h"
#include "BKE_report.h"
#include "BKE_scene.h"

#include "DEG_depsgraph_query.h"

#include "BLI_utildefines.h"

#include "WM_api.h"
#include "WM_types.h"

#include "DNA_gpencil_modifier_types.h"
#include "DNA_gpencil_types.h"
#include "DNA_scene_types.h"

#include "UI_resources.h"

#include "ED_lineart.h"

#include "lineart_intern.h"

static void clear_strokes(Object *ob, GpencilModifierData *md, int frame)
{
  if (md->type != eGpencilModifierType_Lineart) {
    return;
  }
  LineartGpencilModifierData *lmd = (LineartGpencilModifierData *)md;
  bGPdata *gpd = ob->data;

  bGPDlayer *gpl = BKE_gpencil_layer_get_by_name(gpd, lmd->target_layer, 1);
  if (!gpl) {
    return;
  }
  bGPDframe *gpf = BKE_gpencil_layer_frame_find(gpl, frame);

  if (!gpf) {
    /* No greasepencil frame found. */
    return;
  }

  BKE_gpencil_layer_frame_delete(gpl, gpf);
}

// TODO use the job system for baking so that the UI doesn't freeze on big bake tasks.
static void bake_strokes(Object *ob, Depsgraph *dg, GpencilModifierData *md, int frame)
{
  if (md->type != eGpencilModifierType_Lineart) {
    return;
  }
  LineartGpencilModifierData *lmd = (LineartGpencilModifierData *)md;
  bGPdata *gpd = ob->data;

  bGPDlayer *gpl = BKE_gpencil_layer_get_by_name(gpd, lmd->target_layer, 1);
  if (!gpl) {
    return;
  }
  bool only_use_existing_gp_frames = false;
  bGPDframe *gpf = (only_use_existing_gp_frames ?
                        BKE_gpencil_layer_frame_find(gpl, frame) :
                        BKE_gpencil_layer_frame_get(gpl, frame, GP_GETFRAME_ADD_NEW));

  if (!gpf) {
    /* No greasepencil frame created or found. */
    return;
  }

  ED_lineart_compute_feature_lines_internal(dg, lmd);

  ED_lineart_gpencil_generate_with_type(
      lmd->render_buffer,
      dg,
      ob,
      gpl,
      gpf,
      lmd->source_type,
      lmd->source_type == LRT_SOURCE_OBJECT ? (void *)lmd->source_object :
                                              (void *)lmd->source_collection,
      lmd->level_start,
      lmd->use_multiple_levels ? lmd->level_end : lmd->level_start,
      lmd->target_material ? BKE_gpencil_object_material_index_get(ob, lmd->target_material) : 0,
      lmd->line_types,
      lmd->transparency_flags,
      lmd->transparency_mask,
      lmd->thickness,
      lmd->opacity,
      lmd->pre_sample_length,
      lmd->source_vertex_group,
      lmd->vgname,
      lmd->flags);

  ED_lineart_destroy_render_data(lmd);
}

static int lineart_gpencil_bake_all_strokes_invoke(bContext *C,
                                                   wmOperator *op,
                                                   const wmEvent *UNUSED(event))
{
  Scene *scene = CTX_data_scene(C);
  Depsgraph *dg = CTX_data_depsgraph_pointer(C);
  int frame;
  int frame_begin = scene->r.sfra;
  int frame_end = scene->r.efra;
  int frame_orig = scene->r.cfra;
  int frame_increment = scene->r.frame_step;
  bool overwrite_frames = true;

  for (frame = frame_begin; frame <= frame_end; frame += frame_increment) {

    BKE_scene_frame_set(scene, frame);
    BKE_scene_graph_update_for_newframe(dg);

    CTX_DATA_BEGIN (C, Object *, ob, visible_objects) {
      if (ob->type != OB_GPENCIL) {
        continue;
      }

      if (overwrite_frames) {
        LISTBASE_FOREACH (GpencilModifierData *, md, &ob->greasepencil_modifiers) {
          clear_strokes(ob, md, frame);
        }
      }

      LISTBASE_FOREACH (GpencilModifierData *, md, &ob->greasepencil_modifiers) {
        bake_strokes(ob, dg, md, frame);
      }
    }
    CTX_DATA_END;
  }

  /* Restore original frame. */
  BKE_scene_frame_set(scene, frame_orig);
  BKE_scene_graph_update_for_newframe(dg);

  BKE_report(op->reports, RPT_INFO, "Line Art baking is complete.");
  WM_event_add_notifier(C, NC_GPENCIL | ND_DATA | NA_EDITED, NULL);

  return OPERATOR_FINISHED;
}

static int lineart_gpencil_bake_all_strokes_exec(bContext *C, wmOperator *UNUSED(op))
{
  Scene *scene = CTX_data_scene(C);

  /* If confirmed in the dialog, then just turn off the master switch upon finished baking. */
  // scene->lineart.flags &= (~LRT_AUTO_UPDATE);

  return OPERATOR_FINISHED;
}

static int lineart_gpencil_bake_strokes_invoke(bContext *C,
                                               wmOperator *op,
                                               const wmEvent *UNUSED(event))
{
  Object *ob = CTX_data_active_object(C);

  Scene *scene = CTX_data_scene(C);
  Depsgraph *dg = CTX_data_depsgraph_pointer(C);
  int frame;
  int frame_begin = scene->r.sfra;
  int frame_end = scene->r.efra;
  int frame_orig = scene->r.cfra;
  int frame_increment = scene->r.frame_step;
  bool overwrite_frames = true;

  for (frame = frame_begin; frame <= frame_end; frame += frame_increment) {

    BKE_scene_frame_set(scene, frame);
    BKE_scene_graph_update_for_newframe(dg);
    if (ob->type != OB_GPENCIL) {
      continue;
    }

    if (overwrite_frames) {
      LISTBASE_FOREACH (GpencilModifierData *, md, &ob->greasepencil_modifiers) {
        clear_strokes(ob, md, frame);
      }
    }

    LISTBASE_FOREACH (GpencilModifierData *, md, &ob->greasepencil_modifiers) {
      bake_strokes(ob, dg, md, frame);
    }
  }

  /* Restore original frame. */
  BKE_scene_frame_set(scene, frame_orig);
  BKE_scene_graph_update_for_newframe(dg);

  BKE_report(op->reports, RPT_INFO, "Line Art baking is complete.");
  WM_event_add_notifier(C, NC_GPENCIL | ND_DATA | NA_EDITED, NULL);

  return OPERATOR_FINISHED;
}

static int lineart_gpencil_bake_strokes_exec(bContext *C, wmOperator *UNUSED(op))
{
  Scene *scene = CTX_data_scene(C);

  return OPERATOR_FINISHED;
}

static int lineart_gpencil_clear_strokes_invoke(bContext *C,
                                                wmOperator *op,
                                                const wmEvent *UNUSED(event))
{
  Object *ob = CTX_data_active_object(C);
  if (ob->type != OB_GPENCIL) {
    return OPERATOR_CANCELLED;
  }
  LISTBASE_FOREACH (GpencilModifierData *, md, &ob->greasepencil_modifiers) {
    if (md->type != eGpencilModifierType_Lineart) {
      continue;
    }
    LineartGpencilModifierData *lmd = (LineartGpencilModifierData *)md;
    bGPdata *gpd = ob->data;

    bGPDlayer *gpl = BKE_gpencil_layer_get_by_name(gpd, lmd->target_layer, 1);
    if (!gpl) {
      continue;
    }
    BKE_gpencil_free_frames(gpl);
  }

  BKE_report(op->reports, RPT_INFO, "Line Art clear layers is complete.");
  WM_event_add_notifier(C, NC_GPENCIL | ND_DATA | NA_EDITED, NULL);
  return OPERATOR_FINISHED;
}

static int lineart_gpencil_clear_strokes_exec(bContext *C, wmOperator *UNUSED(op))
{
  Scene *scene = CTX_data_scene(C);

  return OPERATOR_FINISHED;
}

static int lineart_gpencil_clear_all_strokes_invoke(bContext *C,
                                                    wmOperator *op,
                                                    const wmEvent *UNUSED(event))
{
  // FIXME ASAN reports a mem leak here in CTX_DATA_BEGIN
  // Yiming: I'm not sure why here it leaks... I'm taling a look.
  CTX_DATA_BEGIN (C, Object *, ob, visible_objects) {
    if (ob->type != OB_GPENCIL) {
      return OPERATOR_CANCELLED;
    }
    LISTBASE_FOREACH (GpencilModifierData *, md, &ob->greasepencil_modifiers) {
      if (md->type != eGpencilModifierType_Lineart) {
        continue;
      }
      LineartGpencilModifierData *lmd = (LineartGpencilModifierData *)md;
      bGPdata *gpd = ob->data;

      bGPDlayer *gpl = BKE_gpencil_layer_get_by_name(gpd, lmd->target_layer, 1);
      if (!gpl) {
        continue;
      }
      BKE_gpencil_free_frames(gpl);
    }
  }
  CTX_DATA_END;

  BKE_report(op->reports, RPT_INFO, "Line Art all clear layers is complete.");
  WM_event_add_notifier(C, NC_GPENCIL | ND_DATA | NA_EDITED, NULL);
  return OPERATOR_FINISHED;
}

static int lineart_gpencil_clear_all_strokes_exec(bContext *C, wmOperator *UNUSED(op))
{
  Scene *scene = CTX_data_scene(C);

  return OPERATOR_FINISHED;
}

/* Bake all line art modifiers on the current object. */
void OBJECT_OT_lineart_bake_strokes(wmOperatorType *ot)
{
  ot->name = "Bake Line Art Strokes";
  ot->description = "Bake Line Art all line art modifier on this object";
  ot->idname = "OBJECT_OT_lineart_bake_strokes";

  ot->invoke = lineart_gpencil_bake_strokes_invoke;
  ot->exec = lineart_gpencil_bake_strokes_exec;
}

/* Bake all lineart objects in the scene. */
void OBJECT_OT_lineart_bake_all_strokes(wmOperatorType *ot)
{
  ot->name = "Bake All Line Art Strokes";
  ot->description = "Bake All Line Art Modifiers In The Scene";
  ot->idname = "OBJECT_OT_lineart_bake_all_strokes";

  ot->invoke = lineart_gpencil_bake_all_strokes_invoke;
  ot->exec = lineart_gpencil_bake_all_strokes_exec;
}

/* clear all line art modifiers on the current object. */
void OBJECT_OT_lineart_clear_strokes(wmOperatorType *ot)
{
  ot->name = "Clear Line Art Strokes";
  ot->description = "Clear Line Art grease pencil strokes for all frames";
  ot->idname = "OBJECT_OT_lineart_clear_strokes";

  ot->invoke = lineart_gpencil_clear_strokes_invoke;
  ot->exec = lineart_gpencil_clear_strokes_exec;
}

/* clear all lineart objects in the scene. */
void OBJECT_OT_lineart_clear_all_strokes(wmOperatorType *ot)
{
  ot->name = "Clear All Line Art Strokes";
  ot->description = "Clear All Line Art Modifiers In The Scene";
  ot->idname = "OBJECT_OT_lineart_clear_all_strokes";

  ot->invoke = lineart_gpencil_clear_all_strokes_invoke;
  ot->exec = lineart_gpencil_clear_all_strokes_exec;
}

void ED_operatortypes_lineart(void)
{
  WM_operatortype_append(OBJECT_OT_lineart_bake_strokes);
  WM_operatortype_append(OBJECT_OT_lineart_bake_all_strokes);
  WM_operatortype_append(OBJECT_OT_lineart_clear_strokes);
  WM_operatortype_append(OBJECT_OT_lineart_clear_all_strokes);
}
