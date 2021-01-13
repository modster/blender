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
 * The Original Code is Copyright (C) 2001-2002 by NaN Holding BV.
 * All rights reserved.
 */

/** \file
 * \ingroup edtransform
 */

#include "DNA_anim_types.h"
#include "DNA_space_types.h"

#include "MEM_guardedalloc.h"

#include "BLI_listbase.h"
#include "BLI_math.h"

#include "BKE_anim_data.h"
#include "BKE_context.h"
#include "BKE_nla.h"

#include "ED_anim_api.h"
#include "ED_markers.h"

#include "WM_api.h"

#include "RNA_access.h"

#include "transform.h"
#include "transform_convert.h"

/** Used for NLA transform (stored in #TransData.extra pointer). */
typedef struct TransDataNla {
  /** ID-block NLA-data is attached to. */
  ID *id;

  /** Original NLA-Track that the strip belongs to. */
  struct NlaTrack *oldTrack;
  /** Current NLA-Track that the strip belongs to. */
  struct NlaTrack *nlt;

  /** NLA-strip this data represents. */
  struct NlaStrip *strip;

  /* dummy values for transform to write in - must have 3 elements... */
  /** start handle. */
  float h1[3];
  /** end handle. */
  float h2[3];

  /** index of track that strip is currently in. */
  int trackIndex;

  /* Important: this index is relative to the initial first track at the start of transforming and
   * thus can be negative when the tracks list grows downward. */
  int signed_track_index;
  /** handle-index: 0 for dummy entry, -1 for start, 1 for end, 2 for both ends. */
  int handle;
} TransDataNla;

static bool is_overlap(const float left_bound_a,
                       const float right_bound_a,
                       const float left_bound_b,
                       const float right_bound_b)
{
  return (left_bound_a < right_bound_b) && (right_bound_a > left_bound_b);
}

static bool nlastrip_is_overlap(NlaStrip *strip_a,
                                float offset_a,
                                NlaStrip *strip_b,
                                float offset_b)
{
  return is_overlap(strip_a->start + offset_a,
                    strip_a->end + offset_a,
                    strip_b->start + offset_b,
                    strip_b->end + offset_b);
}

/** Assumes strips to shuffle are tagged with NLASTRIP_FLAG_FIX_LOCATION.
 *
 * \returns The total sided offset that results in no overlaps between tagged strips and non-tagged
 * strips.
 */
static float transdata_get_time_shuffle_offset_side(ListBase *trans_datas, const bool shuffle_left)
{
  float total_offset = 0;

  float offset;
  do {
    offset = 0;

    LISTBASE_FOREACH (LinkData *, link, trans_datas) {
      TransDataNla *trans_data = (TransDataNla *)link->data;
      NlaStrip *xformed_strip = trans_data->strip;

      LISTBASE_FOREACH (NlaStrip *, non_xformed_strip, &trans_data->nlt->strips) {
        if (non_xformed_strip->flag & NLASTRIP_FLAG_FIX_LOCATION) {
          continue;
        }

        /* Allow overlap with transitions. */
        if (non_xformed_strip->type & NLASTRIP_TYPE_TRANSITION) {
          continue;
        }

        if (!nlastrip_is_overlap(non_xformed_strip, 0, xformed_strip, total_offset)) {
          continue;
        }

        if (shuffle_left) {
          offset = min(offset, non_xformed_strip->start - (xformed_strip->end + total_offset));
        }
        else {
          offset = max(offset, non_xformed_strip->end - (xformed_strip->start + total_offset));
        }
      }
    }

    total_offset += offset;
  } while (!IS_EQF(offset, 0.0f));

  return total_offset;
}

/** Assumes strips to shuffle are tagged with NLASTRIP_FLAG_FIX_LOCATION.
 *
 * \returns The minimal total signed offset that results in no overlaps between tagged strips and
 * non-tagged strips.
 */
static float transdata_get_time_shuffle_offset(ListBase *trans_datas)
{
  const float offset_left = transdata_get_time_shuffle_offset_side(trans_datas, true);
  const float offset_right = transdata_get_time_shuffle_offset_side(trans_datas, false);

  if (fabs(offset_left) < offset_right) {
    return offset_left;
  }
  else {
    return offset_right;
  }
}

/** Assumes all of given trans_datas are part of the same ID.
 *
 * \param r_total_offset: The minimal total signed offset that results in valid strip track-moves
 * for all strips from \a trans_datas.
 *
 * \returns true if \a r_total_offset results in a valid offset, false if no solution exists in the
 * desired direction.
 */
static bool transdata_get_track_shuffle_offset_side(ListBase *trans_datas,
                                                    const bool shuffle_down,
                                                    int *r_total_offset)
{
  *r_total_offset = 0;
  if (BLI_listbase_is_empty(trans_datas)) {
    return false;
  }

  ListBase *tracks = &BKE_animdata_from_id(
                          ((TransDataNla *)((LinkData *)trans_datas->first)->data)->id)
                          ->nla_tracks;

  int offset;
  do {
    offset = 0;

    LISTBASE_FOREACH (LinkData *, link, trans_datas) {
      TransDataNla *trans_data = (TransDataNla *)link->data;
      NlaStrip *xformed_strip = trans_data->strip;

      NlaTrack *dst_track = BLI_findlink(tracks, trans_data->trackIndex + *r_total_offset);

      /** Cannot keep moving strip in given track direction. No solution. */
      if (dst_track == NULL) {
        return false;
      }

      /** Shuffle only if track is locked or library override. */
      if (((dst_track->flag & NLATRACK_PROTECTED) == 0) &&
          !BKE_nlatrack_is_nonlocal_in_liboverride(trans_data->id, dst_track)) {
        continue;
      }

      if (shuffle_down) {
        offset = -1;
      }
      else {
        offset = 1;
      }
      break;
    }

    *r_total_offset += offset;
  } while (offset != 0);

  return true;
}

/** Assumes all of given trans_datas are part of the same ID.
 *
 * \param r_track_offset: The minimal total signed offset that results in valid strip track-moves
 * for all strips from \a trans_datas.
 *
 * \returns true if \a r_track_offset results in a valid offset, false if no solution exists in
 * either direction.
 */
static bool transdata_get_track_shuffle_offset(ListBase *trans_datas, int *r_track_offset)
{
  int offset_down = 0;
  const bool down_valid = transdata_get_track_shuffle_offset_side(trans_datas, true, &offset_down);

  int offset_up = 0;
  const bool up_valid = transdata_get_track_shuffle_offset_side(trans_datas, false, &offset_up);

  if (down_valid && up_valid) {
    if (fabs(offset_down) < offset_up) {
      *r_track_offset = offset_down;
    }
    else {
      *r_track_offset = offset_up;
    }
  }
  else if (down_valid) {
    *r_track_offset = offset_down;
  }
  else if (up_valid) {
    *r_track_offset = offset_up;
  }

  return down_valid || up_valid;
}

/* -------------------------------------------------------------------- */
/** \name NLA Transform Creation
 *
 * \{ */

void createTransNlaData(bContext *C, TransInfo *t)
{
  Scene *scene = t->scene;
  SpaceNla *snla = NULL;
  TransData *td = NULL;
  TransDataNla *tdn = NULL;

  bAnimContext ac;
  ListBase anim_data = {NULL, NULL};
  bAnimListElem *ale;
  int filter;

  int count = 0;

  TransDataContainer *tc = TRANS_DATA_CONTAINER_FIRST_SINGLE(t);

  /* determine what type of data we are operating on */
  if (ANIM_animdata_get_context(C, &ac) == 0) {
    return;
  }
  snla = (SpaceNla *)ac.sl;

  /* filter data */
  filter = (ANIMFILTER_DATA_VISIBLE | ANIMFILTER_LIST_VISIBLE | ANIMFILTER_FOREDIT);
  ANIM_animdata_filter(&ac, &anim_data, filter, ac.data, ac.datatype);

  /* which side of the current frame should be allowed */
  if (t->mode == TFM_TIME_EXTEND) {
    t->frame_side = transform_convert_frame_side_dir_get(t, (float)CFRA);
  }
  else {
    /* normal transform - both sides of current frame are considered */
    t->frame_side = 'B';
  }

  /* loop 1: count how many strips are selected (consider each strip as 2 points) */
  for (ale = anim_data.first; ale; ale = ale->next) {
    NlaTrack *nlt = (NlaTrack *)ale->data;
    NlaStrip *strip;

    /* make some meta-strips for chains of selected strips */
    BKE_nlastrips_make_metas(&nlt->strips, 1);

    /* only consider selected strips */
    for (strip = nlt->strips.first; strip; strip = strip->next) {
      /* TODO: we can make strips have handles later on. */
      /* transition strips can't get directly transformed */
      if (strip->type != NLASTRIP_TYPE_TRANSITION) {
        if (strip->flag & NLASTRIP_FLAG_SELECT) {
          if (FrameOnMouseSide(t->frame_side, strip->start, (float)CFRA)) {
            count++;
          }
          if (FrameOnMouseSide(t->frame_side, strip->end, (float)CFRA)) {
            count++;
          }
        }
      }
    }
  }

  /* stop if trying to build list if nothing selected */
  if (count == 0) {
    /* clear temp metas that may have been created but aren't needed now
     * because they fell on the wrong side of CFRA
     */
    for (ale = anim_data.first; ale; ale = ale->next) {
      NlaTrack *nlt = (NlaTrack *)ale->data;
      BKE_nlastrips_clear_metas(&nlt->strips, 0, 1);
    }

    /* cleanup temp list */
    ANIM_animdata_freelist(&anim_data);
    return;
  }

  /* allocate memory for data */
  tc->data_len = count;

  tc->data = MEM_callocN(tc->data_len * sizeof(TransData), "TransData(NLA Editor)");
  td = tc->data;
  tc->custom.type.data = tdn = MEM_callocN(tc->data_len * sizeof(TransDataNla),
                                           "TransDataNla (NLA Editor)");
  tc->custom.type.use_free = true;

  /* loop 2: build transdata array */
  for (ale = anim_data.first; ale; ale = ale->next) {
    /* only if a real NLA-track */
    if (ale->type == ANIMTYPE_NLATRACK) {
      AnimData *adt = ale->adt;
      NlaTrack *nlt = (NlaTrack *)ale->data;
      NlaStrip *strip;

      /* only consider selected strips */
      for (strip = nlt->strips.first; strip; strip = strip->next) {
        /* TODO: we can make strips have handles later on. */
        /* transition strips can't get directly transformed */
        if (strip->type != NLASTRIP_TYPE_TRANSITION) {
          if (strip->flag & NLASTRIP_FLAG_SELECT) {
            /* our transform data is constructed as follows:
             * - only the handles on the right side of the current-frame get included
             * - td structs are transform-elements operated on by the transform system
             *   and represent a single handle. The storage/pointer used (val or loc) depends on
             *   whether we're scaling or transforming. Ultimately though, the handles
             *   the td writes to will simply be a dummy in tdn
             * - for each strip being transformed, a single tdn struct is used, so in some
             *   cases, there will need to be 1 of these tdn elements in the array skipped...
             */
            float center[3], yval;

            /* firstly, init tdn settings */
            tdn->id = ale->id;
            tdn->oldTrack = tdn->nlt = nlt;
            tdn->strip = strip;
            tdn->trackIndex = BLI_findindex(&adt->nla_tracks, nlt);
            tdn->signed_track_index = tdn->trackIndex;

            yval = (float)(tdn->trackIndex * NLACHANNEL_STEP(snla));

            tdn->h1[0] = strip->start;
            tdn->h1[1] = yval;
            tdn->h2[0] = strip->end;
            tdn->h2[1] = yval;

            center[0] = (float)CFRA;
            center[1] = yval;
            center[2] = 0.0f;

            /* set td's based on which handles are applicable */
            if (FrameOnMouseSide(t->frame_side, strip->start, (float)CFRA)) {
              /* just set tdn to assume that it only has one handle for now */
              tdn->handle = -1;

              /* now, link the transform data up to this data */
              if (ELEM(t->mode, TFM_TRANSLATION, TFM_TIME_EXTEND)) {
                td->loc = tdn->h1;
                copy_v3_v3(td->iloc, tdn->h1);

                /* store all the other gunk that is required by transform */
                copy_v3_v3(td->center, center);
                memset(td->axismtx, 0, sizeof(td->axismtx));
                td->axismtx[2][2] = 1.0f;

                td->ext = NULL;
                td->val = NULL;

                td->flag |= TD_SELECTED;
                td->dist = 0.0f;

                unit_m3(td->mtx);
                unit_m3(td->smtx);
              }
              else {
                /* time scaling only needs single value */
                td->val = &tdn->h1[0];
                td->ival = tdn->h1[0];
              }

              td->extra = tdn;
              td++;
            }
            if (FrameOnMouseSide(t->frame_side, strip->end, (float)CFRA)) {
              /* if tdn is already holding the start handle,
               * then we're doing both, otherwise, only end */
              tdn->handle = (tdn->handle) ? 2 : 1;

              /* now, link the transform data up to this data */
              if (ELEM(t->mode, TFM_TRANSLATION, TFM_TIME_EXTEND)) {
                td->loc = tdn->h2;
                copy_v3_v3(td->iloc, tdn->h2);

                /* store all the other gunk that is required by transform */
                copy_v3_v3(td->center, center);
                memset(td->axismtx, 0, sizeof(td->axismtx));
                td->axismtx[2][2] = 1.0f;

                td->ext = NULL;
                td->val = NULL;

                td->flag |= TD_SELECTED;
                td->dist = 0.0f;

                unit_m3(td->mtx);
                unit_m3(td->smtx);
              }
              else {
                /* time scaling only needs single value */
                td->val = &tdn->h2[0];
                td->ival = tdn->h2[0];
              }

              td->extra = tdn;
              td++;
            }

            /* If both handles were used, skip the next tdn (i.e. leave it blank)
             * since the counting code is dumb.
             * Otherwise, just advance to the next one.
             */
            if (tdn->handle == 2) {
              tdn += 2;
            }
            else {
              tdn++;
            }
          }
        }
      }
    }
  }

  /* cleanup temp list */
  ANIM_animdata_freelist(&anim_data);
}

/* helper for recalcData() - for NLA Editor transforms */
void recalcData_nla(TransInfo *t)
{
  SpaceNla *snla = (SpaceNla *)t->area->spacedata.first;
  Scene *scene = t->scene;
  double secf = FPS;
  int i;

  const bool is_translating = ELEM(t->mode, TFM_TRANSLATION);

  TransDataContainer *tc = TRANS_DATA_CONTAINER_FIRST_SINGLE(t);
  TransDataNla *tdn = tc->custom.type.data;

  /* For each strip we've got, perform some additional validation of the values
   * that got set before using RNA to set the value (which does some special
   * operations when setting these values to make sure that everything works ok).
   */
  for (i = 0; i < tc->data_len; i++, tdn++) {
    NlaStrip *strip = tdn->strip;
    short pExceeded, nExceeded, iter;
    int delta_y1, delta_y2;

    /* if this tdn has no handles, that means it is just a dummy that should be skipped */
    if (tdn->handle == 0) {
      continue;
    }
    strip->flag &= ~NLASTRIP_FLAG_FIX_LOCATION;

    /* set refresh tags for objects using this animation,
     * BUT only if realtime updates are enabled
     */
    if ((snla->flag & SNLA_NOREALTIMEUPDATES) == 0) {
      ANIM_id_update(CTX_data_main(t->context), tdn->id);
    }

    /* if canceling transform, just write the values without validating, then move on */
    if (t->state == TRANS_CANCEL) {
      /* clear the values by directly overwriting the originals, but also need to restore
       * endpoints of neighboring transition-strips
       */

      /* start */
      strip->start = tdn->h1[0];

      if ((strip->prev) && (strip->prev->type == NLASTRIP_TYPE_TRANSITION)) {
        strip->prev->end = tdn->h1[0];
      }

      /* end */
      strip->end = tdn->h2[0];

      if ((strip->next) && (strip->next->type == NLASTRIP_TYPE_TRANSITION)) {
        strip->next->start = tdn->h2[0];
      }

      /* flush transforms to child strips (since this should be a meta) */
      BKE_nlameta_flush_transforms(strip);

      /* restore to original track (if needed) */
      if (tdn->oldTrack != tdn->nlt) {
        /* Just append to end of list for now,
         * since strips get sorted in special_aftertrans_update(). */
        BLI_remlink(&tdn->nlt->strips, strip);
        BLI_addtail(&tdn->oldTrack->strips, strip);
      }

      continue;
    }

    const bool nlatrack_isliboverride = BKE_nlatrack_is_nonlocal_in_liboverride(tdn->id, tdn->nlt);
    const bool allow_overlap = !nlatrack_isliboverride && is_translating;
    if (allow_overlap) {
      /** Reorder strips for proper nla stack evaluation while dragging. */
      while (strip->prev != NULL && tdn->h1[0] < strip->prev->start) {
        BLI_listbase_swaplinks(&tdn->nlt->strips, strip, strip->prev);
      }
      while (strip->next != NULL && tdn->h1[0] > strip->next->start) {
        BLI_listbase_swaplinks(&tdn->nlt->strips, strip, strip->next);
      }
    }
    else {
      /* firstly, check if the proposed transform locations would overlap with any neighboring
       * strips (barring transitions) which are absolute barriers since they are not being moved
       *
       * this is done as a iterative procedure (done 5 times max for now)
       */
      for (iter = 0; iter < 5; iter++) {
        pExceeded = ((strip->prev) && (strip->prev->type != NLASTRIP_TYPE_TRANSITION) &&
                     (tdn->h1[0] < strip->prev->end));
        nExceeded = ((strip->next) && (strip->next->type != NLASTRIP_TYPE_TRANSITION) &&
                     (tdn->h2[0] > strip->next->start));

        if ((pExceeded && nExceeded) || (iter == 4)) {
          /* both endpoints exceeded (or iteration ping-pong'd meaning that we need a compromise)
           * - Simply crop strip to fit within the bounds of the strips bounding it
           * - If there were no neighbors, clear the transforms
           *   (make it default to the strip's current values).
           */
          if (strip->prev && strip->next) {
            tdn->h1[0] = strip->prev->end;
            tdn->h2[0] = strip->next->start;
          }
          else {
            tdn->h1[0] = strip->start;
            tdn->h2[0] = strip->end;
          }
        }
        else if (nExceeded) {
          /* move backwards */
          float offset = tdn->h2[0] - strip->next->start;

          tdn->h1[0] -= offset;
          tdn->h2[0] -= offset;
        }
        else if (pExceeded) {
          /* more forwards */
          float offset = strip->prev->end - tdn->h1[0];

          tdn->h1[0] += offset;
          tdn->h2[0] += offset;
        }
        else { /* all is fine and well */
          break;
        }
      }
    }

    /* handle auto-snapping
     * NOTE: only do this when transform is still running, or we can't restore
     */
    if (t->state != TRANS_CANCEL) {
      switch (snla->autosnap) {
        case SACTSNAP_FRAME: /* snap to nearest frame */
        case SACTSNAP_STEP:  /* frame step - this is basically the same,
                              * since we don't have any remapping going on */
        {
          tdn->h1[0] = floorf(tdn->h1[0] + 0.5f);
          tdn->h2[0] = floorf(tdn->h2[0] + 0.5f);
          break;
        }

        case SACTSNAP_SECOND: /* snap to nearest second */
        case SACTSNAP_TSTEP:  /* second step - this is basically the same,
                               * since we don't have any remapping going on */
        {
          /* This case behaves differently from the rest, since lengths of strips
           * may not be multiples of a second. If we just naively resize adjust
           * the handles, things may not work correctly. Instead, we only snap
           * the first handle, and move the other to fit.
           *
           * FIXME: we do run into problems here when user attempts to negatively
           *        scale the strip, as it then just compresses down and refuses
           *        to expand out the other end.
           */
          float h1_new = (float)(floor(((double)tdn->h1[0] / secf) + 0.5) * secf);
          float delta = h1_new - tdn->h1[0];

          tdn->h1[0] = h1_new;
          tdn->h2[0] += delta;
          break;
        }

        case SACTSNAP_MARKER: /* snap to nearest marker */
        {
          tdn->h1[0] = (float)ED_markers_find_nearest_marker_time(&t->scene->markers, tdn->h1[0]);
          tdn->h2[0] = (float)ED_markers_find_nearest_marker_time(&t->scene->markers, tdn->h2[0]);
          break;
        }
      }
    }

    if (allow_overlap) {
      /* Directly flush. */
      strip->start = tdn->h1[0];
      strip->end = tdn->h2[0];
    }
    else {
      /* Use RNA to write the values to ensure that constraints on these are obeyed
       * (e.g. for transition strips, the values are taken from the neighbors)
       *
       * NOTE: we write these twice to avoid truncation errors which can arise when
       * moving the strips a large distance using numeric input T33852.
       *
       * This also results in transition boundaries updating real-time and prevents
       * transitions from being deleted due to invalid start/end frame.
       */
      PointerRNA strip_ptr;
      RNA_pointer_create(NULL, &RNA_NlaStrip, strip, &strip_ptr);

      RNA_float_set(&strip_ptr, "frame_start", tdn->h1[0]);
      RNA_float_set(&strip_ptr, "frame_end", tdn->h2[0]);

      RNA_float_set(&strip_ptr, "frame_start", tdn->h1[0]);
      RNA_float_set(&strip_ptr, "frame_end", tdn->h2[0]);
    }

    /* flush transforms to child strips (since this should be a meta) */
    BKE_nlameta_flush_transforms(strip);

    /* Now, check if we need to try and move track:
     * - we need to calculate both,
     *   as only one may have been altered by transform if only 1 handle moved.
     */
    /* In LibOverride case, we cannot move strips across tracks that come from the linked data. */
    const bool id_is_liboverride = ID_IS_OVERRIDE_LIBRARY(tdn->id);
    if (nlatrack_isliboverride) {
      continue;
    }

    delta_y1 = ((int)tdn->h1[1] / NLACHANNEL_STEP(snla) - tdn->signed_track_index);
    delta_y2 = ((int)tdn->h2[1] / NLACHANNEL_STEP(snla) - tdn->signed_track_index);

    /* Move strip into track in the requested direction. */
    if (delta_y1 || delta_y2) {
      NlaTrack *track;
      int delta = (delta_y2) ? delta_y2 : delta_y1;
      int n;

      AnimData *anim_data = BKE_animdata_from_id(tdn->id);
      ListBase *nla_tracks = &anim_data->nla_tracks;

      NlaTrack *old_track = tdn->nlt;
      NlaTrack *dst_track = NULL;

      /** Calculate the total new tracks needed.
       *
       * Determine dst_track, which will end up being NULL, the last library override
       * track, or a normal local track. The first two cases lead to delta_new_tracks!=0.
       * The last case leads to delta_new_tracks==0.
       */
      int delta_new_tracks = delta;
      dst_track = old_track;
      {
        while (delta_new_tracks < 0) {
          dst_track = dst_track->prev;
          if (dst_track == NULL || BKE_nlatrack_is_nonlocal_in_liboverride(tdn->id, dst_track)) {
            break;
          }
          delta_new_tracks++;
        }

        /** We assume all library tracks are grouped at the bottom of the nla stack. Thus, no need
         * to check for them when moving tracks upward. */
        while (delta_new_tracks > 0) {
          dst_track = dst_track->next;
          if (dst_track == NULL) {
            break;
          }
          delta_new_tracks--;
        }
      }

      /** Auto-grow track list. */
      {
        for (int i = 0; i < -delta_new_tracks; i++) {
          NlaTrack *new_track = BKE_nlatrack_new();
          new_track->flag |= NLATRACK_TEMPORARILY_ADDED;

          BKE_nlatrack_insert_before(
              nla_tracks, (NlaTrack *)nla_tracks->first, new_track, id_is_liboverride);
          dst_track = new_track;
        }

        for (int i = 0; i < delta_new_tracks; i++) {
          NlaTrack *new_track = BKE_nlatrack_new();
          new_track->flag |= NLATRACK_TEMPORARILY_ADDED;

          BKE_nlatrack_insert_after(
              nla_tracks, (NlaTrack *)nla_tracks->last, new_track, id_is_liboverride);
          dst_track = new_track;
        }
      }

      /** Move strip from old_track to dst_track. */
      if (dst_track != old_track) {
        BKE_nlatrack_remove_strip(old_track, strip);
        BKE_nlastrips_add_strip(&dst_track->strips, strip, false);

        tdn->nlt = dst_track;
        tdn->signed_track_index += delta;
        tdn->trackIndex = BLI_findindex(nla_tracks, dst_track);
      }
    }

    if (tdn->nlt->flag & NLATRACK_PROTECTED) {
      strip->flag |= NLASTRIP_FLAG_FIX_LOCATION;
    }

    /** Flag overlaps with adjacent strips.
     *
     * Since the strips are re-ordered as they're transformed, we only have to check adjacent
     * strips for overlap instead of all of them. */
    {
      NlaStrip *adj_strip = strip->prev;
      if (adj_strip != NULL && !(adj_strip->flag & NLASTRIP_FLAG_SELECT) &&
          nlastrip_is_overlap(strip, 0, adj_strip, 0)) {
        strip->flag |= NLASTRIP_FLAG_FIX_LOCATION;
      }

      adj_strip = strip->next;
      if (adj_strip != NULL && !(adj_strip->flag & NLASTRIP_FLAG_SELECT) &&
          nlastrip_is_overlap(strip, 0, adj_strip, 0)) {
        strip->flag |= NLASTRIP_FLAG_FIX_LOCATION;
      }
    }
  }
}
/** \} */

/* -------------------------------------------------------------------- */
/** \name Special After Transform NLA
 * \{ */

typedef struct IDGroupedTransData {
  struct IDGroupedTransData *next, *prev;

  ID *id;
  ListBase trans_datas;
} IDGroupedTransData;

void special_aftertrans_update__nla(bContext *C, TransInfo *t)
{
  bAnimContext ac;

  /* initialize relevant anim-context 'context' data */
  if (ANIM_animdata_get_context(C, &ac) == 0) {
    return;
  }

  if (!ac.datatype) {
    return;
  }

  TransDataContainer *tc = TRANS_DATA_CONTAINER_FIRST_SINGLE(t);
  TransDataNla *first_trans_data = tc->custom.type.data;
  const bool is_translating = ELEM(t->mode, TFM_TRANSLATION);

  /** Shuffle transformed strips. */
  if (is_translating) {

    /** Element: (IDGroupedTransData*) */
    ListBase grouped_trans_datas = {NULL, NULL};

    /** Flag all non-library-override transformed strips so we can distinguish them when
     * shuffling.
     *
     * Group trans_datas by ID so shuffling is unique per ID.
     */
    {
      TransDataNla *tdn = first_trans_data;
      for (int i = 0; i < tc->data_len; i++, tdn++) {

        /* Skip dummy handles. */
        if (tdn->handle == 0) {
          continue;
        }

        /* For strips within library override tracks, don't do any shuffling at all. Unsure how
         * library overrides should behave so, for now, they're treated as mostly immutable. */
        if ((tdn->nlt->flag & NLATRACK_OVERRIDELIBRARY_LOCAL) == 0) {
          continue;
        }

        tdn->strip->flag |= NLASTRIP_FLAG_FIX_LOCATION;

        IDGroupedTransData *dst_group = NULL;
        /* Find dst_group with matching ID. */
        LISTBASE_FOREACH (IDGroupedTransData *, group, &grouped_trans_datas) {
          if (group->id == tdn->id) {
            dst_group = group;
            break;
          }
        }
        if (dst_group == NULL) {
          dst_group = MEM_callocN(sizeof(IDGroupedTransData), __func__);
          dst_group->id = tdn->id;
          BLI_addhead(&grouped_trans_datas, dst_group);
        }

        BLI_addtail(&dst_group->trans_datas, BLI_genericNodeN(tdn));
      }
    }

    /** Apply shuffling. */
    LISTBASE_FOREACH (IDGroupedTransData *, group, &grouped_trans_datas) {
      ListBase *trans_datas = &group->trans_datas;

      /** Apply vertical shuffle. */
      int minimum_track_offset = 0;
      const bool track_offset_valid = transdata_get_track_shuffle_offset(trans_datas,
                                                                         &minimum_track_offset);
      /** Debug assert to ensure strips preserved their relative track offsets from eachother and
       * none were compressed. Otherwise, no amount of vertical shuffling is a solution.
       *
       * This is considered a bug. */
      BLI_assert(track_offset_valid);

      if (minimum_track_offset != 0) {
        ListBase *tracks = &BKE_animdata_from_id(group->id)->nla_tracks;

        LISTBASE_FOREACH (LinkData *, link, trans_datas) {
          TransDataNla *trans_data = (TransDataNla *)link->data;

          trans_data->trackIndex = trans_data->trackIndex + minimum_track_offset;
          NlaTrack *dst_track = BLI_findlink(tracks, trans_data->trackIndex);

          NlaStrip *strip = trans_data->strip;
          BKE_nlatrack_remove_strip(trans_data->nlt, strip);
          BKE_nlatrack_add_strip(dst_track, strip);

          trans_data->nlt = dst_track;
        }
      }

      /** Apply horizontal shuffle. */
      const float minimum_time_offset = transdata_get_time_shuffle_offset(trans_datas);
      LISTBASE_FOREACH (LinkData *, link, trans_datas) {
        TransDataNla *trans_data = (TransDataNla *)link->data;
        NlaStrip *strip = trans_data->strip;

        strip->start += minimum_time_offset;
        strip->end += minimum_time_offset;
        BKE_nlameta_flush_transforms(strip);
      }
    }

    /** Memory cleanup. */
    LISTBASE_FOREACH (IDGroupedTransData *, group, &grouped_trans_datas) {
      BLI_freelistN(&group->trans_datas);
    }
    BLI_freelistN(&grouped_trans_datas);
  }

  /** Clear NLASTRIP_FLAG_FIX_LOCATION flag. */
  TransDataNla *tdn = first_trans_data;
  for (int i = 0; i < tc->data_len; i++, tdn++) {
    if (tdn->strip == NULL) {
      continue;
    }

    tdn->strip->flag &= ~NLASTRIP_FLAG_FIX_LOCATION;
  }

  ListBase anim_data = {NULL, NULL};
  bAnimListElem *ale;
  short filter = (ANIMFILTER_DATA_VISIBLE | ANIMFILTER_FOREDIT);

  /* get channels to work on */
  ANIM_animdata_filter(&ac, &anim_data, filter, ac.data, ac.datatype);

  for (ale = anim_data.first; ale; ale = ale->next) {
    NlaTrack *nlt = (NlaTrack *)ale->data;

    /* make sure strips are in order again */
    BKE_nlatrack_sort_strips(nlt);

    /* remove the temp metas */
    BKE_nlastrips_clear_metas(&nlt->strips, 0, 1);
  }

  /* free temp memory */
  ANIM_animdata_freelist(&anim_data);

  /** Truncate temporarily added tracks. */
  {
    filter = (ANIMFILTER_DATA_VISIBLE | ANIMFILTER_ANIMDATA);
    ANIM_animdata_filter(&ac, &anim_data, filter, ac.data, ac.datatype);

    for (ale = anim_data.first; ale; ale = ale->next) {
      ListBase *nla_tracks = &ale->adt->nla_tracks;

      /** Remove top tracks that weren't necessary. */
      LISTBASE_FOREACH_BACKWARD_MUTABLE (NlaTrack *, track, nla_tracks) {
        if (!(track->flag & NLATRACK_TEMPORARILY_ADDED)) {
          break;
        }
        if (track->strips.first != NULL) {
          break;
        }
        BKE_nlatrack_remove_and_free(nla_tracks, track, true);
      }

      /** Remove bottom tracks that weren't necessary. */
      LISTBASE_FOREACH_MUTABLE (NlaTrack *, track, nla_tracks) {
        /** Library override tracks are the first N tracks. They're never temporary and determine
         * where we start removing temporaries.*/
        if ((track->flag & NLATRACK_OVERRIDELIBRARY_LOCAL) == 0) {
          continue;
        }
        if (!(track->flag & NLATRACK_TEMPORARILY_ADDED)) {
          break;
        }
        if (track->strips.first != NULL) {
          break;
        }
        BKE_nlatrack_remove_and_free(nla_tracks, track, true);
      }

      /** Clear temporary flag. */
      LISTBASE_FOREACH_MUTABLE (NlaTrack *, track, nla_tracks) {
        track->flag &= ~NLATRACK_TEMPORARILY_ADDED;
      }
    }

    ANIM_animdata_freelist(&anim_data);
  }

  /* perform after-transfrom validation */
  ED_nla_postop_refresh(&ac);
}

/** \} */
