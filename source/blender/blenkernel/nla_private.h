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
 * The Original Code is Copyright (C) 2009 Blender Foundation, Joshua Leung
 * All rights reserved.
 */

/** \file
 * \ingroup bke
 */

#pragma once

#include "BLI_bitmap.h"
#include "BLI_ghash.h"
#include "RNA_types.h"

#ifdef __cplusplus
extern "C" {
#endif

struct AnimationEvalContext;

/* --------------- NLA Evaluation DataTypes ----------------------- */

/* used for list of strips to accumulate at current time */
typedef struct NlaEvalStrip {
  struct NlaEvalStrip *next, *prev;

  NlaTrack *track; /* track that this strip belongs to */
  NlaStrip *strip; /* strip that's being used */

  short track_index; /* the index of the track within the list */
  short strip_mode;  /* which end of the strip are we looking at */

  float strip_time; /* time at which which strip is being evaluated */
} NlaEvalStrip;

/* NlaEvalStrip->strip_mode */
enum eNlaEvalStrip_StripMode {
  /* standard evaluation */
  NES_TIME_BEFORE = -1,
  NES_TIME_WITHIN,
  NES_TIME_AFTER,

  /* transition-strip evaluations */
  NES_TIME_TRANSITION_START,
  NES_TIME_TRANSITION_END,
};

struct NlaEvalChannel;
struct NlaEvalData;

/* Unique channel key for GHash. */
typedef struct NlaEvalChannelKey {
  struct PointerRNA ptr;
  struct PropertyRNA *prop;
} NlaEvalChannelKey;

/* Bitmask of array indices touched by actions. */
typedef struct NlaValidMask {
  BLI_bitmap *ptr;
  BLI_bitmap buffer[sizeof(uint64_t) / sizeof(BLI_bitmap)];
} NlaValidMask;

/* Set of property values for blending. */
typedef struct NlaEvalChannelSnapshot {
  struct NlaEvalChannel *channel;

  /** Only used for keyframe remapping for inverting through a transition. Bit set if raw value was
   * written so we know where the channel value comes from (whether it's the property's default or
   * from the strip). */
  NlaValidMask raw_value_sampled;

  /** Only used for keyframe remapping to know whether channel value can be further inverted to
   * solve for tweak strip value. */
  NlaValidMask invertible;

  int length;   /* Number of values in the property. */
  bool is_base; /* Base snapshot of the channel. */

  float values[]; /* Item values. */
  /* Memory over-allocated to provide space for values. */
} NlaEvalChannelSnapshot;

/* NlaEvalChannel->mix_mode */
enum eNlaEvalChannel_MixMode {
  NEC_MIX_ADD,
  NEC_MIX_MULTIPLY,
  NEC_MIX_QUATERNION,
  NEC_MIX_AXIS_ANGLE,
};

/* Temp channel for accumulating data from NLA for a single property.
 * Handles array properties as a unit to allow intelligent blending. */
typedef struct NlaEvalChannel {
  struct NlaEvalChannel *next, *prev;
  struct NlaEvalData *owner;

  /* Original RNA path string and property key. */
  const char *rna_path;
  NlaEvalChannelKey key;

  int index;
  bool is_array;
  bool in_blend;
  char mix_mode;

  struct NlaEvalChannel *next_blend;
  NlaEvalChannelSnapshot *blend_snapshot;

  /* Mask of array items controlled by NLA. */
  NlaValidMask domain;

  /* Base set of values. */
  NlaEvalChannelSnapshot base_snapshot;
  /* Memory over-allocated to provide space for base_snapshot.values. */
} NlaEvalChannel;

/* Set of values for all channels. */
typedef struct NlaEvalSnapshot {
  /* Snapshot this one defaults to. */
  struct NlaEvalSnapshot *base;

  int size;
  NlaEvalChannelSnapshot **channels;
} NlaEvalSnapshot;

/* Set of all channels covered by NLA. */
typedef struct NlaEvalData {
  ListBase channels;

  /* Mapping of paths and NlaEvalChannelKeys to channels. */
  GHash *path_hash;
  GHash *key_hash;

  /* Base snapshot. */
  int num_channels;
  NlaEvalSnapshot base_snapshot;

  /* Evaluation result shapshot. */
  NlaEvalSnapshot eval_snapshot;
} NlaEvalData;

typedef struct NlaPoseChannelPreBlend {
  // char *bone_name;
  NlaEvalChannel *location;
  NlaEvalChannel *rotation;
  NlaEvalChannel *scale;

} NlaPoseChannelPreBlend;

/* Information about the currently edited strip and ones below it for keyframing. */
typedef struct NlaKeyframingContext {
  struct NlaKeyframingContext *next, *prev;

  /* AnimData for which this context was built. */
  struct AnimData *adt;

  /* Data of the currently edited strip (copy, or fake strip for the main action). */
  NlaStrip strip;
  NlaEvalStrip *eval_strip;
  /** Storage for the non-stashed action as a strip. Only used for keyframing. */
  NlaStrip nonuser_act_strip;

  /* Strips above tweaked strip. */
  ListBase upper_estrips;
  /* Evaluated NLA stack below the tweak strip. */
  NlaEvalData lower_nla_channels;
} NlaKeyframingContext;

/* --------------- NLA Functions (not to be used as a proper API) ----------------------- */

/* convert from strip time <-> global time */
float nlastrip_get_frame(NlaStrip *strip, float cframe, short mode);

/* --------------- NLA Evaluation (very-private stuff) ----------------------- */
/* these functions are only defined here to avoid problems with the order
 * in which they get defined. */

NlaEvalStrip *nlastrips_ctime_get_strip(ListBase *list,
                                        ListBase *strips,
                                        short index,
                                        const struct AnimationEvalContext *anim_eval_context,
                                        const bool flush_to_original);
void nlastrip_evaluate(PointerRNA *ptr,
                       NlaEvalData *channels,
                       ListBase *modifiers,
                       NlaEvalStrip *nes,
                       NlaEvalSnapshot *snapshot,
                       const struct AnimationEvalContext *anim_eval_context,
                       const bool flush_to_original,
                       bool allow_alloc_channels);
void nlastrip_evaluate_invert_get_lower_values(
    PointerRNA *ptr,
    NlaEvalData *upper_eval_data,
    ListBase *modifiers,
    NlaEvalStrip *nes,
    NlaEvalSnapshot *snapshot,
    const struct AnimationEvalContext *anim_eval_context);
void nlastrip_evaluate_raw_value(PointerRNA *ptr,
                                 NlaEvalData *upper_eval_data,
                                 ListBase *modifiers,
                                 NlaEvalStrip *nes,
                                 NlaEvalSnapshot *snapshot,
                                 const struct AnimationEvalContext *anim_eval_context,
                                 short *r_blendmode,
                                 float *r_influence);
void nladata_flush_channels(PointerRNA *ptr,
                            NlaEvalData *channels,
                            NlaEvalSnapshot *snapshot,
                            const bool flush_to_original);

#ifdef __cplusplus
}
#endif
