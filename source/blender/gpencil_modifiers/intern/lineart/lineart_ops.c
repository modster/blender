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

#ifdef LINEART_WITH_BAKE

static int lineart_gpencil_update_strokes_exec(bContext *C, wmOperator *UNUSED(op))
{
  Depsgraph *dg = CTX_data_depsgraph_pointer(C);

  BLI_spin_lock(&lineart_share.lock_loader);

  ED_lineart_compute_feature_lines_background(dg, 0);

  /* Wait for loading finish. */
  BLI_spin_lock(&lineart_share.lock_loader);
  BLI_spin_unlock(&lineart_share.lock_loader);

  WM_event_add_notifier(C, NC_GPENCIL | ND_DATA | NA_EDITED | ND_SPACE_PROPERTIES, NULL);

  return OPERATOR_FINISHED;
}

static int lineart_gpencil_bake_strokes_invoke(bContext *C,
                                               wmOperator *op,
                                               const wmEvent *UNUSED(event))
{
  Scene *scene = CTX_data_scene(C);
  Depsgraph *dg = CTX_data_depsgraph_pointer(C);
  int frame;
  int frame_begin = ((lineart->flags & LRT_BAKING_FINAL_RANGE) ? MAX2(scene->r.sfra, 1) :
                                                                 lineart->baking_preview_start);
  int frame_end = ((lineart->flags & LRT_BAKING_FINAL_RANGE) ? scene->r.efra :
                                                               lineart->baking_preview_end);
  int frame_total = frame_end - frame_begin;
  int frame_orig = scene->r.cfra;
  int frame_increment = ((lineart->flags & LRT_BAKING_KEYFRAMES_ONLY) ?
                             1 :
                             (lineart->baking_skip + 1));
  LineartGpencilModifierData *lmd;
  LineartRenderBuffer *rb;
  int use_types;
  bool frame_updated;

  /* Needed for progress report. */
  lineart_share.wm = CTX_wm_manager(C);
  lineart_share.main_window = CTX_wm_window(C);

  for (frame = frame_begin; frame <= frame_end; frame += frame_increment) {

    frame_updated = false;

    FOREACH_COLLECTION_VISIBLE_OBJECT_RECURSIVE_BEGIN (
        scene->master_collection, ob, DAG_EVAL_RENDER) {

      int cleared = 0;
      if (ob->type != OB_GPENCIL) {
        continue;
      }

      LISTBASE_FOREACH (GpencilModifierData *, md, &ob->greasepencil_modifiers) {
        if (md->type != eGpencilModifierType_Lineart) {
          continue;
        }
        lmd = (LineartGpencilModifierData *)md;
        bGPdata *gpd = ob->data;
        bGPDlayer *gpl = BKE_gpencil_layer_get_by_name(gpd, lmd->target_layer, 1);
        bGPDframe *gpf = ((lineart->flags & LRT_BAKING_KEYFRAMES_ONLY) ?
                              BKE_gpencil_layer_frame_find(gpl, frame) :
                              BKE_gpencil_layer_frame_get(gpl, frame, GP_GETFRAME_ADD_NEW));

        if (!gpf) {
          continue; /* happens when it's keyframe only. */
        }

        if (!frame_updated) {
          /* Reset flags. LRT_SYNC_IGNORE prevent any line art modifiers run calculation
           * function when depsgraph calls for modifier evalurates. */
          ED_lineart_modifier_sync_flag_set(LRT_SYNC_IGNORE, false);
          ED_lineart_calculation_flag_set(LRT_RENDER_IDLE);

          BKE_scene_frame_set(scene, frame);
          BKE_scene_graph_update_for_newframe(dg);

          ED_lineart_update_render_progress(
              (int)((float)(frame - frame_begin) / frame_total * 100), NULL);

          BLI_spin_lock(&lineart_share.lock_loader);
          ED_lineart_compute_feature_lines_background(dg, 0);

          /* Wait for loading finish. */
          BLI_spin_lock(&lineart_share.lock_loader);
          BLI_spin_unlock(&lineart_share.lock_loader);

          while (!ED_lineart_modifier_sync_flag_check(LRT_SYNC_FRESH) ||
                 !ED_lineart_calculation_flag_check(LRT_RENDER_FINISHED)) {
            /* Wait till it's done. */
          }

          ED_lineart_chain_clear_picked_flag(lineart_share.render_buffer);

          frame_updated = true;
        }

        /* Clear original frame. */
        if ((scene->lineart.flags & LRT_GPENCIL_OVERWRITE) && (!cleared)) {
          BKE_gpencil_layer_frame_delete(gpl, gpf);
          gpf = BKE_gpencil_layer_frame_get(gpl, frame, GP_GETFRAME_ADD_NEW);
          cleared = 1;
        }

        rb = lineart_share.render_buffer;

        if (rb->fuzzy_everything) {
          use_types = LRT_EDGE_FLAG_CONTOUR;
        }
        else if (rb->fuzzy_intersections) {
          use_types = lmd->line_types | LRT_EDGE_FLAG_INTERSECTION;
        }
        else {
          use_types = lmd->line_types;
        }

        ED_lineart_gpencil_generate_with_type(
            dg,
            ob,
            gpl,
            gpf,
            lmd->source_type,
            lmd->source_type == LRT_SOURCE_OBJECT ? (void *)lmd->source_object :
                                                    (void *)lmd->source_collection,
            lmd->level_start,
            lmd->use_multiple_levels ? lmd->level_end : lmd->level_start,
            lmd->target_material ?
                BKE_gpencil_object_material_index_get(ob, lmd->target_material) :
                0,
            use_types,
            lmd->transparency_flags,
            lmd->transparency_mask,
            lmd->thickness,
            lmd->opacity,
            lmd->pre_sample_length,
            lmd->source_vertex_group,
            lmd->vgname,
            lmd->flags);
      }
    }
    FOREACH_COLLECTION_VISIBLE_OBJECT_RECURSIVE_END;
  }

  /* Restore original frame. */
  BKE_scene_frame_set(scene, frame_orig);
  BKE_scene_graph_update_for_newframe(dg);

  ED_lineart_modifier_sync_flag_set(LRT_SYNC_IDLE, false);
  ED_lineart_calculation_flag_set(LRT_RENDER_FINISHED);

  BKE_report(op->reports, RPT_INFO, "Line Art baking is complete.");
  WM_operator_confirm_message_ex(C,
                                 op,
                                 "Line Art baking is complete.",
                                 ICON_MOD_WIREFRAME,
                                 "Disable Line Art master switch",
                                 WM_OP_EXEC_REGION_WIN);

  WM_event_add_notifier(C, NC_GPENCIL | ND_DATA | NA_EDITED | ND_SPACE_PROPERTIES, NULL);

  ED_lineart_update_render_progress(100, NULL);

  return OPERATOR_FINISHED;
}

static int lineart_gpencil_bake_strokes_exec(bContext *C, wmOperator *UNUSED(op))
{
  Scene *scene = CTX_data_scene(C);

  /* If confirmed in the dialog, then just turn off the master switch upon finished baking. */
  scene->lineart.flags &= (~LRT_AUTO_UPDATE);

  return OPERATOR_FINISHED;
}

/* Blocking 1 frame update. */
void SCENE_OT_lineart_update_strokes(wmOperatorType *ot)
{
  ot->name = "Update Line Art Strokes";
  ot->description = "Update strokes for Line Art grease pencil targets";
  ot->idname = "SCENE_OT_lineart_update_strokes";

  ot->exec = lineart_gpencil_update_strokes_exec;
}

/* All frames in range. */
void SCENE_OT_lineart_bake_strokes(wmOperatorType *ot)
{
  ot->name = "Bake Line Art Strokes";
  ot->description = "Bake Line Art into grease pencil strokes for all frames";
  ot->idname = "SCENE_OT_lineart_bake_strokes";

  ot->invoke = lineart_gpencil_bake_strokes_invoke;
  ot->exec = lineart_gpencil_bake_strokes_exec;
}

#endif

void ED_operatortypes_lineart(void)
{
  // WM_operatortype_append(SCENE_OT_lineart_update_strokes);
  // WM_operatortype_append(SCENE_OT_lineart_bake_strokes);
}
