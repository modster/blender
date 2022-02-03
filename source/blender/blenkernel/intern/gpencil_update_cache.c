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

/** \file
 * \ingroup bke
 */

#include <stdio.h>

#include "BKE_gpencil_update_cache.h"

#include "BLI_dlrbTree.h"
#include "BLI_listbase.h"

#include "BKE_gpencil.h"

#include "DNA_gpencil_types.h"
#include "DNA_userdef_types.h"

#include "MEM_guardedalloc.h"

static GPencilUpdateCache *update_cache_alloc(int index, int flag, void *data)
{
  GPencilUpdateCache *new_cache = MEM_callocN(sizeof(GPencilUpdateCache), __func__);
  new_cache->children = BLI_dlrbTree_new();
  new_cache->flag = flag;
  new_cache->index = index;
  new_cache->data = data;

  return new_cache;
}

static short cache_node_compare(void *node, void *data)
{
  int index_a = ((GPencilUpdateCacheNode *)node)->cache->index;
  int index_b = ((GPencilUpdateCache *)data)->index;
  if (index_a == index_b) {
    return 0;
  }
  return index_a < index_b ? 1 : -1;
}

static DLRBT_Node *cache_node_alloc(void *data)
{
  GPencilUpdateCacheNode *new_node = MEM_callocN(sizeof(GPencilUpdateCacheNode), __func__);
  new_node->cache = ((GPencilUpdateCache *)data);
  return (DLRBT_Node *)new_node;
}

static void cache_node_free(void *node);

static void update_cache_free(GPencilUpdateCache *cache)
{
  if (cache->children != NULL) {
    BLI_dlrbTree_free(cache->children, cache_node_free);
    MEM_freeN(cache->children);
  }
  MEM_freeN(cache);
}

static void cache_node_free(void *node)
{
  GPencilUpdateCache *cache = ((GPencilUpdateCacheNode *)node)->cache;
  if (cache != NULL) {
    update_cache_free(cache);
  }
  MEM_freeN(node);
}

static void cache_node_update(void *node, void *data)
{
  GPencilUpdateCache *update_cache = ((GPencilUpdateCacheNode *)node)->cache;
  GPencilUpdateCache *new_update_cache = (GPencilUpdateCache *)data;

  /* If the new cache is already "covered" by the current cache, just free it and return. */
  if (new_update_cache->flag <= update_cache->flag) {
    update_cache_free(new_update_cache);
    return;
  }

  update_cache->data = new_update_cache->data;
  update_cache->flag = new_update_cache->flag;

  /* In case the new cache does a full update, remove its children since they will be all
   * updated by this cache. */
  if (new_update_cache->flag == GP_UPDATE_NODE_FULL_COPY && update_cache->children != NULL) {
    /* We don't free the tree itself here, because we just want to clear the children, not delete
     * the whole node. */
    BLI_dlrbTree_free(update_cache->children, cache_node_free);
  }

  update_cache_free(new_update_cache);
}

static void update_cache_node_create_ex(GPencilUpdateCache *root_cache,
                                        void *data,
                                        int gpl_index,
                                        int gpf_index,
                                        int gps_index,
                                        bool full_copy)
{
  if (root_cache->flag == GP_UPDATE_NODE_FULL_COPY) {
    /* Entire data-block has to be recaculated, e.g. nothing else needs to be added to the cache.
     */
    return;
  }

  const int node_flag = full_copy ? GP_UPDATE_NODE_FULL_COPY : GP_UPDATE_NODE_LIGHT_COPY;

  if (gpl_index == -1) {
    root_cache->data = (bGPdata *)data;
    root_cache->flag = node_flag;
    if (full_copy) {
      /* Entire data-block has to be recaculated, remove all caches of "lower" elements. */
      BLI_dlrbTree_free(root_cache->children, cache_node_free);
    }
    return;
  }

  const bool is_layer_update_node = (gpf_index == -1);
  /* If the data pointer in GPencilUpdateCache is NULL, this element is not actually cached
   * and does not need to be updated, but we do need the index to find elements that are in
   * levels below. E.g. if a stroke needs to be updated, the frame it is in would not hold a
   * pointer to it's data. */
  GPencilUpdateCache *gpl_cache = update_cache_alloc(
      gpl_index,
      is_layer_update_node ? node_flag : GP_UPDATE_NODE_NO_COPY,
      is_layer_update_node ? (bGPDlayer *)data : NULL);
  GPencilUpdateCacheNode *gpl_node = (GPencilUpdateCacheNode *)BLI_dlrbTree_add(
      root_cache->children, cache_node_compare, cache_node_alloc, cache_node_update, gpl_cache);

  BLI_dlrbTree_linkedlist_sync(root_cache->children);
  if (gpl_node->cache->flag == GP_UPDATE_NODE_FULL_COPY || is_layer_update_node) {
    return;
  }

  const bool is_frame_update_node = (gps_index == -1);
  GPencilUpdateCache *gpf_cache = update_cache_alloc(
      gpf_index,
      is_frame_update_node ? node_flag : GP_UPDATE_NODE_NO_COPY,
      is_frame_update_node ? (bGPDframe *)data : NULL);
  GPencilUpdateCacheNode *gpf_node = (GPencilUpdateCacheNode *)BLI_dlrbTree_add(
      gpl_node->cache->children,
      cache_node_compare,
      cache_node_alloc,
      cache_node_update,
      gpf_cache);

  BLI_dlrbTree_linkedlist_sync(gpl_node->cache->children);
  if (gpf_node->cache->flag == GP_UPDATE_NODE_FULL_COPY || is_frame_update_node) {
    return;
  }

  GPencilUpdateCache *gps_cache = update_cache_alloc(gps_index, node_flag, (bGPDstroke *)data);
  BLI_dlrbTree_add(gpf_node->cache->children,
                   cache_node_compare,
                   cache_node_alloc,
                   cache_node_update,
                   gps_cache);

  BLI_dlrbTree_linkedlist_sync(gpf_node->cache->children);
}

static void update_cache_node_create(
    bGPdata *gpd, bGPDlayer *gpl, bGPDframe *gpf, bGPDstroke *gps, bool full_copy)
{
  if (gpd == NULL) {
    return;
  }

  GPencilUpdateCache *root_cache = gpd->runtime.update_cache;
  if (root_cache == NULL) {
    gpd->runtime.update_cache = update_cache_alloc(0, GP_UPDATE_NODE_NO_COPY, NULL);
    root_cache = gpd->runtime.update_cache;
  }

  if (root_cache->flag == GP_UPDATE_NODE_FULL_COPY) {
    /* Entire data-block has to be recaculated, e.g. nothing else needs to be added to the cache.
     */
    return;
  }

  const int gpl_index = (gpl != NULL) ? BLI_findindex(&gpd->layers, gpl) : -1;
  const int gpf_index = (gpl != NULL && gpf != NULL) ? BLI_findindex(&gpl->frames, gpf) : -1;
  const int gps_index = (gpf != NULL && gps != NULL) ? BLI_findindex(&gpf->strokes, gps) : -1;

  void *data = gps;
  if (!data) {
    data = gpf;
  }
  if (!data) {
    data = gpl;
  }
  if (!data) {
    data = gpd;
  }

  update_cache_node_create_ex(root_cache, data, gpl_index, gpf_index, gps_index, full_copy);
}

static void gpencil_traverse_update_cache_ex(GPencilUpdateCache *parent_cache,
                                             GPencilUpdateCacheTraverseSettings *ts,
                                             int depth,
                                             void *user_data)
{
  if (BLI_listbase_is_empty((ListBase *)parent_cache->children)) {
    return;
  }

  LISTBASE_FOREACH (GPencilUpdateCacheNode *, cache_node, parent_cache->children) {
    GPencilUpdateCache *cache = cache_node->cache;

    GPencilUpdateCacheIter_Cb cb = ts->update_cache_cb[depth];
    if (cb != NULL) {
      bool skip = cb(cache, user_data);
      if (skip) {
        continue;
      }
    }

    gpencil_traverse_update_cache_ex(cache, ts, depth + 1, user_data);
  }
}

typedef struct GPencilUpdateCacheDuplicateTraverseData {
  GPencilUpdateCache *new_cache;
  int gpl_index;
  int gpf_index;
} GPencilUpdateCacheDuplicateTraverseData;

static bool gpencil_duplicate_update_cache_layer_cb(GPencilUpdateCache *cache, void *user_data)
{
  GPencilUpdateCacheDuplicateTraverseData *td = (GPencilUpdateCacheDuplicateTraverseData *)
      user_data;

  if (cache->flag == GP_UPDATE_NODE_FULL_COPY) {
    /* Do a full copy of the layer. */
    bGPDlayer *gpl = (bGPDlayer *)cache->data;
    bGPDlayer *gpl_new = BKE_gpencil_layer_duplicate(gpl, true, true);
    update_cache_node_create_ex(td->new_cache, gpl_new, cache->index, -1, -1, true);
    return true;
  }
  else if (cache->flag == GP_UPDATE_NODE_LIGHT_COPY) {
    bGPDlayer *gpl = (bGPDlayer *)cache->data;
    bGPDlayer *gpl_new = (bGPDlayer *)MEM_dupallocN(gpl);

    gpl_new->prev = gpl_new->next = NULL;
    BLI_listbase_clear(&gpl_new->frames);
    BLI_listbase_clear(&gpl_new->mask_layers);
    update_cache_node_create_ex(td->new_cache, gpl_new, cache->index, -1, -1, false);
  }
  td->gpl_index = cache->index;
  return false;
}

static bool gpencil_duplicate_update_cache_frame_cb(GPencilUpdateCache *cache, void *user_data)
{
  GPencilUpdateCacheDuplicateTraverseData *td = (GPencilUpdateCacheDuplicateTraverseData *)
      user_data;
  if (cache->flag == GP_UPDATE_NODE_FULL_COPY) {
    bGPDframe *gpf = (bGPDframe *)cache->data;
    bGPDframe *gpf_new = BKE_gpencil_frame_duplicate(gpf, true);
    update_cache_node_create_ex(td->new_cache, gpf_new, td->gpl_index, cache->index, -1, true);
    return true;
  }
  else if (cache->flag == GP_UPDATE_NODE_LIGHT_COPY) {
    bGPDframe *gpf = (bGPDframe *)cache->data;
    bGPDframe *gpf_new = MEM_dupallocN(gpf);
    gpf_new->prev = gpf_new->next = NULL;
    BLI_listbase_clear(&gpf_new->strokes);
    update_cache_node_create_ex(td->new_cache, gpf_new, td->gpl_index, cache->index, -1, false);
  }
  td->gpf_index = cache->index;
  return false;
}

static bool gpencil_duplicate_update_cache_stroke_cb(GPencilUpdateCache *cache, void *user_data)
{
  GPencilUpdateCacheDuplicateTraverseData *td = (GPencilUpdateCacheDuplicateTraverseData *)
      user_data;

  if (cache->flag == GP_UPDATE_NODE_FULL_COPY) {
    bGPDstroke *gps = (bGPDstroke *)cache->data;
    bGPDstroke *gps_new = BKE_gpencil_stroke_duplicate(gps, true, true);
    update_cache_node_create_ex(
        td->new_cache, gps_new, td->gpl_index, td->gpf_index, cache->index, true);
  }
  else if (cache->flag == GP_UPDATE_NODE_LIGHT_COPY) {
    bGPDstroke *gps = (bGPDstroke *)cache->data;
    bGPDstroke *gps_new = MEM_dupallocN(gps);

    gps_new->prev = gps_new->next = NULL;
    gps_new->points = NULL;
    gps_new->triangles = NULL;
    gps_new->dvert = NULL;
    gps_new->editcurve = NULL;

    update_cache_node_create_ex(
        td->new_cache, gps_new, td->gpl_index, td->gpf_index, cache->index, false);
  }
  return true;
}

static bool gpencil_free_update_cache_layer_cb(GPencilUpdateCache *cache, void *UNUSED(user_data))
{
  if (cache->flag == GP_UPDATE_NODE_FULL_COPY) {
    BKE_gpencil_free_frames(cache->data);
    BKE_gpencil_free_layer_masks(cache->data);
  }
  if (cache->data) {
    MEM_freeN(cache->data);
  }
  return cache->flag == GP_UPDATE_NODE_FULL_COPY;
}

static bool gpencil_free_update_cache_frame_cb(GPencilUpdateCache *cache, void *UNUSED(user_data))
{
  if (cache->flag == GP_UPDATE_NODE_FULL_COPY) {
    BKE_gpencil_free_strokes(cache->data);
  }
  if (cache->data) {
    MEM_freeN(cache->data);
  }
  return cache->flag == GP_UPDATE_NODE_FULL_COPY;
}

static bool gpencil_free_update_cache_stroke_cb(GPencilUpdateCache *cache, void *UNUSED(user_data))
{
  if (cache->flag == GP_UPDATE_NODE_FULL_COPY) {
    BKE_gpencil_free_stroke(cache->data);
  }
  return cache->flag == GP_UPDATE_NODE_FULL_COPY;
}

static bool gpencil_print_update_cache_layer_cb(GPencilUpdateCache *cache, void *UNUSED(user_data))
{
  printf("  - Layer: %s | Index: %d | Flag: %d | Tagged Frames: %d\n",
         (cache->data ? ((bGPDlayer *)cache->data)->info : "N/A"),
         cache->index,
         cache->flag,
         BLI_listbase_count((ListBase *)cache->children));
  return cache->flag == GP_UPDATE_NODE_FULL_COPY;
}

static bool gpencil_print_update_cache_frame_cb(GPencilUpdateCache *cache, void *UNUSED(user_data))
{
  printf("  - Layer: %s | Index: %d | Flag: %d | Tagged Frames: %d\n",
         (cache->data ? ((bGPDlayer *)cache->data)->info : "N/A"),
         cache->index,
         cache->flag,
         BLI_listbase_count((ListBase *)cache->children));
  return cache->flag == GP_UPDATE_NODE_FULL_COPY;
}

static bool gpencil_print_update_cache_stroke_cb(GPencilUpdateCache *cache,
                                                 void *UNUSED(user_data))
{
  printf("     - Stroke Index: %d | | Flag: %d\n", cache->index, cache->flag);
  return cache->flag == GP_UPDATE_NODE_FULL_COPY;
}

/* -------------------------------------------------------------------- */
/** \name Update Cache API
 *
 * \{ */

GPencilUpdateCache *BKE_gpencil_create_update_cache(void *data, bool full_copy)
{
  return update_cache_alloc(
      0, full_copy ? GP_UPDATE_NODE_FULL_COPY : GP_UPDATE_NODE_LIGHT_COPY, data);
}

void BKE_gpencil_traverse_update_cache(GPencilUpdateCache *cache,
                                       GPencilUpdateCacheTraverseSettings *ts,
                                       void *user_data)
{
  gpencil_traverse_update_cache_ex(cache, ts, 0, user_data);
}

void BKE_gpencil_tag_full_update(bGPdata *gpd, bGPDlayer *gpl, bGPDframe *gpf, bGPDstroke *gps)
{
  if (U.experimental.use_gpencil_update_cache) {
    update_cache_node_create(gpd, gpl, gpf, gps, true);
  }
}

void BKE_gpencil_tag_light_update(bGPdata *gpd, bGPDlayer *gpl, bGPDframe *gpf, bGPDstroke *gps)
{
  if (U.experimental.use_gpencil_update_cache) {
    update_cache_node_create(gpd, gpl, gpf, gps, false);
  }
}

GPencilUpdateCache *BKE_gpencil_duplicate_update_cache_and_data(GPencilUpdateCache *gpd_cache)
{
  GPencilUpdateCache *new_cache = update_cache_alloc(0, gpd_cache->flag, NULL);
  bGPdata *gpd_new = NULL;
  if (gpd_cache->flag == GP_UPDATE_NODE_FULL_COPY) {
    BKE_gpencil_data_duplicate(NULL, gpd_cache->data, &gpd_new);
    new_cache->data = gpd_new;
    return new_cache;
  }
  else if (gpd_cache->flag == GP_UPDATE_NODE_LIGHT_COPY) {
    gpd_new = MEM_dupallocN(gpd_cache->data);

    /* Clear all the pointers, since they shouldn't store anything. */
    BLI_listbase_clear(&gpd_new->layers);
    BLI_listbase_clear(&gpd_new->vertex_group_names);
    gpd_new->adt = NULL;
    gpd_new->mat = NULL;
    gpd_new->runtime.update_cache = NULL;

    new_cache->data = gpd_new;
  }

  GPencilUpdateCacheTraverseSettings ts = {{gpencil_duplicate_update_cache_layer_cb,
                                            gpencil_duplicate_update_cache_frame_cb,
                                            gpencil_duplicate_update_cache_stroke_cb}};

  GPencilUpdateCacheDuplicateTraverseData td = {
      .new_cache = new_cache,
      .gpl_index = -1,
      .gpf_index = -1,
  };

  BKE_gpencil_traverse_update_cache(gpd_cache, &ts, &td);
  return new_cache;
}

/**
 *  Return true if any of the branches in gpd_cache_b are "strictly greater than" the branches in
 * gpd_cache_a, e.g. one of them contains more data than their counterpart.
 */
bool BKE_gpencil_compare_update_caches(GPencilUpdateCache *gpd_cache_a,
                                       GPencilUpdateCache *gpd_cache_b)
{
  if (gpd_cache_b->flag == GP_UPDATE_NODE_FULL_COPY) {
    return gpd_cache_a->flag != GP_UPDATE_NODE_FULL_COPY;
  }
  if (gpd_cache_a->flag == GP_UPDATE_NODE_FULL_COPY) {
    return false;
  }

  LISTBASE_FOREACH (GPencilUpdateCacheNode *, node_b, gpd_cache_b->children) {
    GPencilUpdateCacheNode *node_a = (GPencilUpdateCacheNode *)BLI_dlrbTree_search_exact(
        gpd_cache_a->children, cache_node_compare, node_b->cache);
    if (node_a == NULL) {
      return true;
    }

    if (BKE_gpencil_compare_update_caches(node_a->cache, node_b->cache)) {
      return true;
    }
  }
  return false;
}

void BKE_gpencil_free_update_cache(bGPdata *gpd)
{
  GPencilUpdateCache *gpd_cache = gpd->runtime.update_cache;
  if (gpd_cache) {
    update_cache_free(gpd_cache);
    gpd->runtime.update_cache = NULL;
  }
  gpd->flag &= ~GP_DATA_UPDATE_CACHE_UNDO_ENCODED;
}

void BKE_gpencil_free_update_cache_and_data(GPencilUpdateCache *gpd_cache)
{
  if (gpd_cache->data != NULL) {
    if (gpd_cache->flag == GP_UPDATE_NODE_FULL_COPY) {
      BKE_gpencil_free_data(gpd_cache->data, true);
      MEM_freeN(gpd_cache->data);
      update_cache_free(gpd_cache);
      return;
    }
    MEM_freeN(gpd_cache->data);
  }

  GPencilUpdateCacheTraverseSettings ts = {{gpencil_free_update_cache_layer_cb,
                                            gpencil_free_update_cache_frame_cb,
                                            gpencil_free_update_cache_stroke_cb}};

  BKE_gpencil_traverse_update_cache(gpd_cache, &ts, NULL);
  update_cache_free(gpd_cache);
}

void BKE_gpencil_print_update_cache(bGPdata *gpd)
{
  GPencilUpdateCache *update_cache = gpd->runtime.update_cache;

  if (update_cache == NULL) {
    printf("No update cache\n");
    return;
  }
  printf("Update Cache:\n");
  printf("- GPdata: %s | Flag: %d | Tagged Layers: %d\n",
         gpd->id.name,
         update_cache->flag,
         BLI_listbase_count((ListBase *)update_cache->children));

  GPencilUpdateCacheTraverseSettings ts = {{gpencil_print_update_cache_layer_cb,
                                            gpencil_print_update_cache_frame_cb,
                                            gpencil_print_update_cache_stroke_cb}};
  BKE_gpencil_traverse_update_cache(update_cache, &ts, NULL);
}

/** \} */
