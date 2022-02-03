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
 * The Original Code is Copyright (C) 2011 Blender Foundation.
 * All rights reserved.
 */

/** \file
 * \ingroup edgpencil
 */

#include <stdlib.h>
#include <string.h>

#include "MEM_guardedalloc.h"

#include "BLI_listbase.h"
#include "DNA_gpencil_types.h"
#include "DNA_listBase.h"
#include "DNA_object_types.h"
#include "DNA_scene_types.h"
#include "DNA_windowmanager_types.h"

#include "BKE_blender_undo.h"
#include "BKE_context.h"
#include "BKE_gpencil.h"
#include "BKE_gpencil_update_cache.h"
#include "BKE_undo_system.h"

#include "ED_gpencil.h"
#include "ED_undo.h"

#include "WM_api.h"
#include "WM_types.h"

#include "DEG_depsgraph.h"

#include "gpencil_intern.h"

typedef struct bGPundonode {
  struct bGPundonode *next, *prev;

  char name[BKE_UNDO_STR_MAX];
  struct bGPdata *gpd;
} bGPundonode;

static ListBase undo_nodes = {NULL, NULL};
static bGPundonode *cur_node = NULL;

int ED_gpencil_session_active(void)
{
  return (BLI_listbase_is_empty(&undo_nodes) == false);
}

int ED_undo_gpencil_step(bContext *C, const int step)
{
  bGPdata **gpd_ptr = NULL, *new_gpd = NULL;

  gpd_ptr = ED_gpencil_data_get_pointers(C, NULL);

  const eUndoStepDir undo_step = (eUndoStepDir)step;
  if (undo_step == STEP_UNDO) {
    if (cur_node->prev) {
      cur_node = cur_node->prev;
      new_gpd = cur_node->gpd;
    }
  }
  else if (undo_step == STEP_REDO) {
    if (cur_node->next) {
      cur_node = cur_node->next;
      new_gpd = cur_node->gpd;
    }
  }

  if (new_gpd) {
    if (gpd_ptr) {
      if (*gpd_ptr) {
        bGPdata *gpd = *gpd_ptr;
        bGPDlayer *gpld;

        BKE_gpencil_free_layers(&gpd->layers);

        /* copy layers */
        BLI_listbase_clear(&gpd->layers);

        LISTBASE_FOREACH (bGPDlayer *, gpl_undo, &gpd->layers) {
          /* make a copy of source layer and its data */
          gpld = BKE_gpencil_layer_duplicate(gpl_undo, true, true);
          BLI_addtail(&gpd->layers, gpld);
        }
      }
    }
    /* drawing batch cache is dirty now */
    DEG_id_tag_update(&new_gpd->id, ID_RECALC_TRANSFORM | ID_RECALC_GEOMETRY);
    new_gpd->flag |= GP_DATA_CACHE_IS_DIRTY;
  }

  WM_event_add_notifier(C, NC_GPENCIL | NA_EDITED, NULL);

  return OPERATOR_FINISHED;
}

void gpencil_undo_init(bGPdata *gpd)
{
  gpencil_undo_push(gpd);
}

static void gpencil_undo_free_node(bGPundonode *undo_node)
{
  /* HACK: animdata wasn't duplicated, so it shouldn't be freed here,
   * or else the real copy will segfault when accessed
   */
  undo_node->gpd->adt = NULL;

  BKE_gpencil_free_data(undo_node->gpd, false);
  MEM_freeN(undo_node->gpd);
}

void gpencil_undo_push(bGPdata *gpd)
{
  bGPundonode *undo_node;

  if (cur_node) {
    /* Remove all undone nodes from stack. */
    undo_node = cur_node->next;

    while (undo_node) {
      bGPundonode *next_node = undo_node->next;

      gpencil_undo_free_node(undo_node);
      BLI_freelinkN(&undo_nodes, undo_node);

      undo_node = next_node;
    }
  }

  /* limit number of undo steps to the maximum undo steps
   * - to prevent running out of memory during **really**
   *   long drawing sessions (triggering swapping)
   */
  /* TODO: Undo-memory constraint is not respected yet,
   * but can be added if we have any need for it. */
  if (U.undosteps && !BLI_listbase_is_empty(&undo_nodes)) {
    /* remove anything older than n-steps before cur_node */
    int steps = 0;

    undo_node = (cur_node) ? cur_node : undo_nodes.last;
    while (undo_node) {
      bGPundonode *prev_node = undo_node->prev;

      if (steps >= U.undosteps) {
        gpencil_undo_free_node(undo_node);
        BLI_freelinkN(&undo_nodes, undo_node);
      }

      steps++;
      undo_node = prev_node;
    }
  }

  /* create new undo node */
  undo_node = MEM_callocN(sizeof(bGPundonode), "gpencil undo node");
  BKE_gpencil_data_duplicate(NULL, gpd, &undo_node->gpd);

  cur_node = undo_node;

  BLI_addtail(&undo_nodes, undo_node);
}

void gpencil_undo_finish(void)
{
  bGPundonode *undo_node = undo_nodes.first;

  while (undo_node) {
    gpencil_undo_free_node(undo_node);
    undo_node = undo_node->next;
  }

  BLI_freelistN(&undo_nodes);

  cur_node = NULL;
}

/* -------------------------------------------------------------------- */
/** \name Implements ED Undo System
 * \{ */

typedef struct GPencilUndoData {
  GPencilUpdateCache *gpd_cache_data;
  /* Scene frame for this step. */
  int cfra;
  /* Store the grease pencil mode we are in. */
  eObjectMode mode;
} GPencilUndoData;

typedef struct GPencilUndoStep {
  UndoStep step;
  GPencilUndoData *undo_data;
} GPencilUndoStep;

static bool change_gpencil_mode(bContext *C, Object *ob, eObjectMode mode)
{
  if (ob->mode == mode) {
    return false;
  }
  bGPdata *gpd = (bGPdata *)ob->data;
  ob->mode = mode;
  ED_gpencil_setup_modes(C, gpd, mode);

  return true;
}

static void gpencil_data_to_undo_data(bGPdata *gpd, GPencilUndoData *gpd_undo_data)
{
  GPencilUpdateCache *update_cache = gpd->runtime.update_cache;

  if (update_cache == NULL) {
    /* Need a full-copy of the grease pencil undo_data. */
    bGPdata *gpd_copy = NULL;
    BKE_gpencil_data_duplicate(NULL, gpd, &gpd_copy);
    gpd_copy->id.session_uuid = gpd->id.session_uuid;

    gpd_undo_data->gpd_cache_data = BKE_gpencil_create_update_cache(gpd_copy, true);
  }
  else {
    gpd_undo_data->gpd_cache_data = BKE_gpencil_duplicate_update_cache_and_data(update_cache);
  }
}

typedef struct tGPencilUpdateCacheUndoTraverseData {
  bGPdata *gpd;
  bGPDlayer *gpl;
  bGPDframe *gpf;
  bGPDstroke *gps;
  int gpl_index;
  int gpf_index;
  int gps_index;
  bool tag_update_cache;
} tGPencilUpdateCacheUndoTraverseData;

static bool gpencil_decode_undo_data_layer_cb(GPencilUpdateCache *gpl_cache, void *user_data)
{
  tGPencilUpdateCacheUndoTraverseData *td = (tGPencilUpdateCacheUndoTraverseData *)user_data;
  td->gpl = BLI_findlinkfrom((Link *)td->gpl, gpl_cache->index - td->gpl_index);
  td->gpl_index = gpl_cache->index;
  bGPDlayer *gpl_new = (bGPDlayer *)gpl_cache->data;

  if (gpl_cache->flag == GP_UPDATE_NODE_FULL_COPY) {
    /* Do a full copy of the layer. */
    bGPDlayer *gpl_next = td->gpl->next;
    BKE_gpencil_layer_delete(td->gpd, td->gpl);

    td->gpl = BKE_gpencil_layer_duplicate(gpl_new, true, true);
    BLI_insertlinkbefore(&td->gpd->layers, gpl_next, td->gpl);

    if (td->tag_update_cache) {
      /* Tag the layer here. */
      BKE_gpencil_tag_full_update(td->gpd, td->gpl, NULL, NULL);
    }
    return true;
  }
  else if (gpl_cache->flag == GP_UPDATE_NODE_LIGHT_COPY) {
    BKE_gpencil_layer_copy_settings(gpl_new, td->gpl);
    if (td->tag_update_cache) {
      BKE_gpencil_tag_light_update(td->gpd, td->gpl, NULL, NULL);
    }
  }

  td->gpf = td->gpl->frames.first;
  td->gpf_index = 0;
  return false;
}

static bool gpencil_decode_undo_data_frame_cb(GPencilUpdateCache *gpf_cache, void *user_data)
{
  tGPencilUpdateCacheUndoTraverseData *td = (tGPencilUpdateCacheUndoTraverseData *)user_data;
  td->gpf = BLI_findlinkfrom((Link *)td->gpf, gpf_cache->index - td->gpf_index);
  td->gpf_index = gpf_cache->index;
  bGPDframe *gpf_new = (bGPDframe *)gpf_cache->data;

  if (gpf_cache->flag == GP_UPDATE_NODE_FULL_COPY) {
    /* Do a full copy of the frame. */
    bGPDframe *gpf_next = td->gpf->next;

    bool update_actframe = (td->gpl->actframe == td->gpf) ? true : false;
    BKE_gpencil_free_strokes(td->gpf);
    BLI_freelinkN(&td->gpl->frames, td->gpf);

    td->gpf = BKE_gpencil_frame_duplicate(gpf_new, true);
    BLI_insertlinkbefore(&td->gpl->frames, gpf_next, td->gpf);

    if (update_actframe) {
      td->gpl->actframe = td->gpf;
    }
    if (td->tag_update_cache) {
      /* Tag the frame here. */
      BKE_gpencil_tag_full_update(td->gpd, td->gpl, td->gpf, NULL);
    }
    return true;
  }
  else if (gpf_cache->flag == GP_UPDATE_NODE_LIGHT_COPY) {
    BKE_gpencil_frame_copy_settings(gpf_new, td->gpf);
    if (td->tag_update_cache) {
      BKE_gpencil_tag_light_update(td->gpd, td->gpl, td->gpf, NULL);
    }
  }

  td->gps = td->gpf->strokes.first;
  td->gps_index = 0;
  return false;
}

static bool gpencil_decode_undo_data_stroke_cb(GPencilUpdateCache *gps_cache, void *user_data)
{
  tGPencilUpdateCacheUndoTraverseData *td = (tGPencilUpdateCacheUndoTraverseData *)user_data;
  td->gps = BLI_findlinkfrom((Link *)td->gps, gps_cache->index - td->gps_index);
  td->gps_index = gps_cache->index;
  bGPDstroke *gps_new = (bGPDstroke *)gps_cache->data;

  if (gps_cache->flag == GP_UPDATE_NODE_FULL_COPY) {
    /* Do a full copy of the stroke. */
    bGPDstroke *gps_next = td->gps->next;

    BLI_remlink(&td->gpf->strokes, td->gps);
    BKE_gpencil_free_stroke(td->gps);

    td->gps = BKE_gpencil_stroke_duplicate(gps_new, true, true);
    BLI_insertlinkbefore(&td->gpf->strokes, gps_next, td->gps);

    if (td->tag_update_cache) {
      /* Tag the stroke here. */
      BKE_gpencil_tag_full_update(td->gpd, td->gpl, td->gpf, td->gps);
    }
  }
  else if (gps_cache->flag == GP_UPDATE_NODE_LIGHT_COPY) {
    BKE_gpencil_stroke_copy_settings(gps_new, td->gps);
    if (td->tag_update_cache) {
      BKE_gpencil_tag_light_update(td->gpd, td->gpl, td->gpf, td->gps);
    }
  }
  return false;
}

static bool gpencil_undo_data_to_gpencil_data(GPencilUndoData *gpd_undo_data,
                                              bGPdata *gpd,
                                              bool tag_gpd_update_cache)
{
  GPencilUpdateCache *update_cache = gpd_undo_data->gpd_cache_data;

  BLI_assert(update_cache != NULL);

  if (update_cache->flag == GP_UPDATE_NODE_FULL_COPY) {
    /* Full-copy. */
    BKE_gpencil_free_data(gpd, true);
    BKE_gpencil_data_duplicate(NULL, update_cache->data, &gpd);
    if (tag_gpd_update_cache) {
      BKE_gpencil_tag_full_update(gpd, NULL, NULL, NULL);
    }
    return true;
  }
  else if (update_cache->flag == GP_UPDATE_NODE_LIGHT_COPY) {
    BKE_gpencil_data_copy_settings(update_cache->data, gpd);
    if (tag_gpd_update_cache) {
      BKE_gpencil_tag_light_update(gpd, NULL, NULL, NULL);
    }
  }

  GPencilUpdateCacheTraverseSettings ts = {{
      gpencil_decode_undo_data_layer_cb,
      gpencil_decode_undo_data_frame_cb,
      gpencil_decode_undo_data_stroke_cb,
  }};

  tGPencilUpdateCacheUndoTraverseData data = {
      .gpd = gpd,
      .gpl = gpd->layers.first,
      .gpf = NULL,
      .gps = NULL,
      .gpl_index = 0,
      .gpf_index = 0,
      .gps_index = 0,
      .tag_update_cache = tag_gpd_update_cache,
  };

  BKE_gpencil_traverse_update_cache(update_cache, &ts, &data);

  return true;
}

static bool gpencil_undosys_poll(bContext *C)
{
  if (!U.experimental.use_gpencil_undo_system) {
    return false;
  }
  bGPdata *gpd = ED_gpencil_data_get_active(C);
  return GPENCIL_ANY_MODE(gpd);
}

static bool gpencil_undosys_step_encode(struct bContext *C,
                                        struct Main *UNUSED(bmain),
                                        UndoStep *us_p)
{
  GPencilUndoStep *us = (GPencilUndoStep *)us_p;

  UndoStack *undo_stack = ED_undo_stack_get();
  Scene *scene = CTX_data_scene(C);
  Object *ob = CTX_data_active_object(C);
  bGPdata *gpd = (bGPdata *)ob->data;

  bool only_frame_changed = false;

  /* TODO: We might need to check if ID_RECALC_ALL is set on the gpd here to test if we need to
   * clear the cache. It might be bad to "start" with some cache and add new update nodes on top.
   */

  /* In case the step we are about to encode would be the first in the gpencil undo system, ensure
   * that we do a full-copy. */
  if (undo_stack->step_active == NULL ||
      undo_stack->step_active->type != BKE_UNDOSYS_TYPE_GPENCIL) {
    BKE_gpencil_tag_full_update(gpd, NULL, NULL, NULL);
  }
  /* If the ID of the grease pencil object was not tagged or the update cache is empty, we assume
   * the data hasn't changed. */
  else if ((gpd->id.recalc & ID_RECALC_ALL) == 0 && gpd->runtime.update_cache == NULL) {
    /* If the previous step is of our undo system, check if the frame changed. */
    if (undo_stack->step_active && undo_stack->step_active->type == BKE_UNDOSYS_TYPE_GPENCIL) {
      GPencilUndoStep *us_prev = (GPencilUndoStep *)undo_stack->step_active;
      /* We want to be able to undo frame changes, so check this here. */
      only_frame_changed = us_prev->undo_data->cfra != scene->r.cfra;
      if (!only_frame_changed) {
        /* If the frame did not change, we don't need to encode anything, return. */
        return false;
      }
    }
    else {
      /* No change (that we want to undo) happend, return. */
      return false;
    }
  }

  /* TODO: Handle this case properly once the update cache is more widly used. We avoid full-copies
   * for now at the expense of to being able to undo them. */
#if 0
  if (!only_frame_changed && gpd->runtime.update_cache == NULL) {
    return false;
  }
#endif

  us->undo_data = MEM_callocN(sizeof(GPencilUndoData), __func__);
  us->undo_data->cfra = scene->r.cfra;
  us->undo_data->mode = ob->mode;

  /* If that step only encodes a frame change (data itself has not changed), return early. */
  if (only_frame_changed) {
    return true;
  }

  gpencil_data_to_undo_data(gpd, us->undo_data);
  gpd->flag |= GP_DATA_UPDATE_CACHE_DISPOSABLE;
  return true;
}

static void gpencil_undosys_step_decode(struct bContext *C,
                                        struct Main *UNUSED(bmain),
                                        UndoStep *us_p,
                                        const eUndoStepDir dir,
                                        bool is_final)
{
  GPencilUndoStep *us = (GPencilUndoStep *)us_p;
  GPencilUndoData *undo_data = us->undo_data;

  Object *ob = CTX_data_active_object(C);
  bGPdata *gpd = (bGPdata *)ob->data;

  if (gpd == NULL) {
    return;
  }

  Scene *scene = CTX_data_scene(C);
  if (undo_data->cfra != scene->r.cfra) {
    scene->r.cfra = undo_data->cfra;
    /* TODO: what if we merged a full copy with a frame change? */
    DEG_id_tag_update(&scene->id, ID_RECALC_AUDIO_MUTE);
    WM_event_add_notifier(C, NC_SCENE | ND_FRAME, NULL);
  }

  if (change_gpencil_mode(C, ob, undo_data->mode)) {
    if (is_final) {
      gpd->flag |= GP_DATA_UPDATE_CACHE_DISPOSABLE;
      DEG_id_tag_update(&gpd->id, ID_RECALC_TRANSFORM | ID_RECALC_GEOMETRY);
      WM_event_add_notifier(C, NC_GPENCIL | ND_DATA | ND_GPENCIL_EDITMODE, NULL);
      WM_event_add_notifier(C, NC_SCENE | ND_MODE, NULL);
    }
    return;
  }

  if (dir == STEP_UNDO) {
    UndoStep *us_iter = us_p;
    BLI_assert(us_iter->next != NULL);
    UndoStep *us_next = us_p->next;

    GPencilUndoData *data_iter = ((GPencilUndoStep *)us_iter)->undo_data;
    GPencilUndoData *data_next = ((GPencilUndoStep *)us_next)->undo_data;

    if (data_next->gpd_cache_data == NULL) {
      return;
    }

    while (
        data_iter->gpd_cache_data == NULL ||
        BKE_gpencil_compare_update_caches(data_iter->gpd_cache_data, data_next->gpd_cache_data)) {
      us_iter = us_iter->prev;
      BLI_assert(us_iter != NULL && us_iter->type == BKE_UNDOSYS_TYPE_GPENCIL);
      data_iter = ((GPencilUndoStep *)us_iter)->undo_data;
    }

    while (us_iter != us_next) {
      if (data_iter->gpd_cache_data != NULL) {
        gpencil_undo_data_to_gpencil_data(data_iter, gpd, true);
      }
      us_iter = us_iter->next;
      data_iter = ((GPencilUndoStep *)us_iter)->undo_data;
    }
  }
  else if (dir == STEP_REDO) {
    if (undo_data->gpd_cache_data == NULL) {
      return;
    }

    gpencil_undo_data_to_gpencil_data(undo_data, gpd, true);
  }
  else {
    BLI_assert_unreachable();
  }

  if (is_final) {
    gpd->flag |= GP_DATA_UPDATE_CACHE_DISPOSABLE;
    DEG_id_tag_update(&gpd->id, ID_RECALC_GEOMETRY);
    WM_event_add_notifier(C, NC_GEOM | ND_DATA, NULL);
  }
}

static void gpencil_undosys_step_free(UndoStep *us_p)
{
  GPencilUndoStep *us = (GPencilUndoStep *)us_p;
  GPencilUndoData *us_data = us->undo_data;

  /**
   * If this undo step is the first, we want to keep its full copy of the grease pencil undo_data
   * (we assume that the first undo step always has this). Otherwise we free the step and its
   * undo_data.
   */
  if ((us_p->prev == NULL || us_p->prev->type != BKE_UNDOSYS_TYPE_GPENCIL) && us_p->next != NULL &&
      us_p->next->type == BKE_UNDOSYS_TYPE_GPENCIL) {
    GPencilUndoStep *us_next = (GPencilUndoStep *)us_p->next;
    GPencilUndoData *us_next_data = us_next->undo_data;
    /* If e.g. a frame change happend, there is no cache so in this case we move the gpd pointer to
     * that step. */
    if (us_next_data->gpd_cache_data == NULL) {
      bGPdata *gpd_copy = us_data->gpd_cache_data->data;
      BLI_assert(gpd_copy != NULL);
      BLI_assert(us_data->gpd_cache_data->flag == GP_UPDATE_NODE_FULL_COPY);

      us_next_data->gpd_cache_data = BKE_gpencil_create_update_cache(gpd_copy, true);
      /* Make sure the gpd_copy is not freed below. */
      us_data->gpd_cache_data->data = NULL;
    }
    /* If the next step does not have a full copy, we need to apply the changes of the next step
     * to our cached gpencil undo_data copy and move it to the next step (it will now be the
     * full-copy). */
    else if (us_next_data->gpd_cache_data->flag != GP_UPDATE_NODE_FULL_COPY) {
      bGPdata *gpd_copy = us_data->gpd_cache_data->data;
      BLI_assert(gpd_copy != NULL);
      BLI_assert(us_data->gpd_cache_data->flag == GP_UPDATE_NODE_FULL_COPY);

      gpencil_undo_data_to_gpencil_data(us_next_data, gpd_copy, false);
      BKE_gpencil_free_update_cache_and_data(us_next_data->gpd_cache_data);

      us_next_data->gpd_cache_data = BKE_gpencil_create_update_cache(gpd_copy, true);

      /* Make sure the gpd_copy is not freed below. */
      us_data->gpd_cache_data->data = NULL;
    }
    else {
      /* If the next step is a full copy, we can safely free the current step (since the last step
       * will be a full-copy). */
    }
  }

  if (us_data->gpd_cache_data) {
    BKE_gpencil_free_update_cache_and_data(us_data->gpd_cache_data);
  }
  MEM_freeN(us_data);
}

void ED_gpencil_undosys_type(UndoType *ut)
{
  ut->name = "Grease Pencil Undo";
  ut->poll = gpencil_undosys_poll;
  ut->step_encode = gpencil_undosys_step_encode;
  ut->step_decode = gpencil_undosys_step_decode;
  ut->step_free = gpencil_undosys_step_free;

  ut->flags = UNDOTYPE_FLAG_NEED_CONTEXT_FOR_ENCODE;

  ut->step_size = sizeof(GPencilUndoStep);
}

/** \} */
