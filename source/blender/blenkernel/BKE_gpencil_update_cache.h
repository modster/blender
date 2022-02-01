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
 */

#pragma once

/** \file
 * \ingroup bke
 */

#ifdef __cplusplus
extern "C" {
#endif

#include "BLI_sys_types.h" /* for bool */

struct DLRBT_Tree;
struct bGPdata;
struct bGPDlayer;
struct bGPDframe;
struct bGPDstroke;
struct GPencilUpdateCache;

/* GPencilUpdateCache.flag */
typedef enum eGPUpdateCacheNodeFlag {
  /* Node is a placeholder (e.g. when only an index is needed). */
  GP_UPDATE_NODE_NO_COPY = -1,
  /* Copy the element as well as all of its content. */
  GP_UPDATE_NODE_FULL_COPY = 0,
  /* Copy only element, not the content. */
  GP_UPDATE_NODE_STRUCT_COPY = 1,
} eGPUpdateCacheNodeFlag;

/**
 *  Cache for what needs to be updated after bGPdata was modified.
 *
 *  Every node holds information about one element that was changed:
 *    - the index of where that element is in the linked-list
 *    - the pointer to the original element in bGPdata
 *  Additionally, nodes also hold other nodes that are one "level" below them.
 *  E.g. a node that represents a change on a bGPDframe could contain a set of
 *  nodes that represent a change on bGPDstrokes.
 *  These nodes are stored in a red-black tree so that they are sorted by their
 *  index to make sure they can be processed in the correct order.
 */
typedef struct GPencilUpdateCache {
  /* Mapping from index to a GPencilUpdateCache struct. */
  struct DLRBT_Tree *children;
  /* eGPUpdateCacheNodeFlag */
  int flag;
  /* Index of the element in the linked-list. */
  int index;
  /* Pointer to one of bGPdata, bGPDLayer, bGPDFrame, bGPDStroke. */
  void *data;
} GPencilUpdateCache;

/* Node structure in the DLRBT_Tree for GPencilUpdateCache mapping. */
typedef struct GPencilUpdateCacheNode {
  /* DLRB tree capabilities. */
  struct GPencilUpdateCacheNode *next, *prev;
  struct GPencilUpdateCacheNode *left, *right;
  struct GPencilUpdateCacheNode *parent;
  char tree_col;

  char _pad[7];
  /* Content of DLRB tree node. */
  GPencilUpdateCache *cache;
} GPencilUpdateCacheNode;

/**
 *
 */
typedef bool (*GPencilUpdateCacheIter_Cb)(struct GPencilUpdateCache *cache, void *user_data);

typedef struct GPencilUpdateCacheTraverseSettings {
  /* Callbacks for the update cache traversal. Callback with index 0 is for layers, 1 for frames
   * and 2 for strokes. */
  GPencilUpdateCacheIter_Cb update_cache_cb[3];
} GPencilUpdateCacheTraverseSettings;

struct GPencilUpdateCache *BKE_gpencil_create_update_cache_data(void *data, bool full_copy);

/**
 * Traverse an update cache and execute callbacks at each level.
 * \param cache: The update cache to traverse.
 * \param ts: The traversal settings. This stores the callbacks that are called at each level.
 * \param user_data: Custom data passed to each callback.
 */
void BKE_gpencil_traverse_update_cache(struct GPencilUpdateCache *cache,
                                       GPencilUpdateCacheTraverseSettings *ts,
                                       void *user_data);

void BKE_gpencil_tag_full_update(struct bGPdata *gpd,
                                 struct bGPDlayer *gpl,
                                 struct bGPDframe *gpf,
                                 struct bGPDstroke *gps);

void BKE_gpencil_tag_struct_update(struct bGPdata *gpd,
                                   struct bGPDlayer *gpl,
                                   struct bGPDframe *gpf,
                                   struct bGPDstroke *gps);

void BKE_gpencil_free_update_cache(struct bGPdata *gpd);

#ifdef __cplusplus
}
#endif
