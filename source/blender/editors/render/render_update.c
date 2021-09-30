/*
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 *
 * The Original Code is Copyright (C) 2009 Blender Foundation.
 * All rights reserved.
 */

/** \file
 * \ingroup edrend
 */

#include <stdlib.h>
#include <string.h>

#include "DNA_cachefile_types.h"
#include "DNA_light_types.h"
#include "DNA_material_types.h"
#include "DNA_node_types.h"
#include "DNA_object_types.h"
#include "DNA_scene_types.h"
#include "DNA_screen_types.h"
#include "DNA_space_types.h"
#include "DNA_view3d_types.h"
#include "DNA_windowmanager_types.h"
#include "DNA_world_types.h"

#include "DRW_engine.h"

#include "BLI_listbase.h"
#include "BLI_threads.h"
#include "BLI_utildefines.h"

#include "BKE_context.h"
#include "BKE_icons.h"
#include "BKE_main.h"
#include "BKE_material.h"
#include "BKE_node.h"
#include "BKE_paint.h"
#include "BKE_scene.h"

#include "RE_engine.h"
#include "RE_pipeline.h"

#include "ED_node.h"
#include "ED_paint.h"
#include "ED_render.h"
#include "ED_view3d.h"

#include "DEG_depsgraph.h"
#include "DEG_depsgraph_query.h"

#include "WM_api.h"

#include <stdio.h>

/* -------------------------------------------------------------------- */
/** \name Render Engines
 * \{ */

/* Update 3D viewport render or draw engine on changes to the scene or view settings. */
void ED_render_view3d_update(Depsgraph *depsgraph,
                             wmWindow *window,
                             ScrArea *area,
                             const bool updated)
{
  Main *bmain = DEG_get_bmain(depsgraph);
  Scene *scene = DEG_get_input_scene(depsgraph);
  ViewLayer *view_layer = DEG_get_input_view_layer(depsgraph);

  LISTBASE_FOREACH (ARegion *, region, &area->regionbase) {
    if (region->regiontype != RGN_TYPE_WINDOW) {
      continue;
    }

    View3D *v3d = area->spacedata.first;

    /* call update if the scene changed, or if the render engine
     * tagged itself for update (e.g. because it was busy at the
     * time of the last update) */
    {
      RenderEngineType *engine_type = ED_view3d_engine_type(scene, v3d->shading.type);
      DRW_notify_view_update((&(DRWUpdateContext){
                                 .bmain = bmain,
                                 .depsgraph = depsgraph,
                                 .scene = scene,
                                 .view_layer = view_layer,
                                 .area = area,
                                 .region = region,
                                 .v3d = v3d,
                                 .engine_type = engine_type,
                                 .window = window,
                             }),
                             updated);
    }
  }
}

/* Update all 3D viewport render and draw engines on changes to the scene.
 * This is called by the dependency graph when it detects changes. */
void ED_render_scene_update(const DEGEditorUpdateContext *update_ctx, const bool updated)
{
  Main *bmain = update_ctx->bmain;
  static bool recursive_check = false;

  /* don't do this render engine update if we're updating the scene from
   * other threads doing e.g. rendering or baking jobs */
  if (!BLI_thread_is_main()) {
    return;
  }

  /* don't call this recursively for frame updates */
  if (recursive_check) {
    return;
  }

  /* Do not call if no WM available, see T42688. */
  if (BLI_listbase_is_empty(&bmain->wm)) {
    return;
  }

  recursive_check = true;

  wmWindowManager *wm = bmain->wm.first;
  LISTBASE_FOREACH (wmWindow *, window, &wm->windows) {
    bScreen *screen = WM_window_get_active_screen(window);

    LISTBASE_FOREACH (ScrArea *, area, &screen->areabase) {
      if (area->spacetype == SPACE_VIEW3D) {
        ED_render_view3d_update(update_ctx->depsgraph, window, area, updated);
      }
    }
  }

  recursive_check = false;
}

void ED_render_engine_area_exit(Main *bmain, ScrArea *area)
{
  /* clear all render engines in this area */
  ARegion *region;
  wmWindowManager *wm = bmain->wm.first;

  if (area->spacetype != SPACE_VIEW3D) {
    return;
  }

  for (region = area->regionbase.first; region; region = region->next) {
    if (region->regiontype != RGN_TYPE_WINDOW || !(region->regiondata)) {
      continue;
    }
    ED_view3d_stop_render_preview(wm, region);
  }
}

void ED_render_engine_changed(Main *bmain, const bool update_scene_data)
{
  /* on changing the render engine type, clear all running render engines */
  for (bScreen *screen = bmain->screens.first; screen; screen = screen->id.next) {
    LISTBASE_FOREACH (ScrArea *, area, &screen->areabase) {
      ED_render_engine_area_exit(bmain, area);
    }
  }
  RE_FreePersistentData(NULL);
  /* Inform all render engines and draw managers. */
  DEGEditorUpdateContext update_ctx = {NULL};
  update_ctx.bmain = bmain;
  for (Scene *scene = bmain->scenes.first; scene; scene = scene->id.next) {
    update_ctx.scene = scene;
    LISTBASE_FOREACH (ViewLayer *, view_layer, &scene->view_layers) {
      /* TDODO(sergey): Iterate over depsgraphs instead? */
      update_ctx.depsgraph = BKE_scene_ensure_depsgraph(bmain, scene, view_layer);
      update_ctx.view_layer = view_layer;
      ED_render_id_flush_update(&update_ctx, &scene->id);
    }
    if (scene->nodetree && update_scene_data) {
      ntreeCompositUpdateRLayers(scene->nodetree);
    }
  }

  /* Update #CacheFiles to ensure that procedurals are properly taken into account. */
  LISTBASE_FOREACH (CacheFile *, cachefile, &bmain->cachefiles) {
    /* Only update cache-files which are set to use a render procedural.
     * We do not use #BKE_cachefile_uses_render_procedural here as we need to update regardless of
     * the current engine or its settings. */
    if (cachefile->use_render_procedural) {
      DEG_id_tag_update(&cachefile->id, ID_RECALC_COPY_ON_WRITE);
      /* Rebuild relations so that modifiers are reconnected to or disconnected from the
       * cache-file. */
      DEG_relations_tag_update(bmain);
    }
  }
}

void ED_render_view_layer_changed(Main *bmain, bScreen *screen)
{
  LISTBASE_FOREACH (ScrArea *, area, &screen->areabase) {
    ED_render_engine_area_exit(bmain, area);
  }
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Updates
 *
 * #ED_render_id_flush_update gets called from #DEG_id_tag_update,
 * to do editor level updates when the ID changes.
 * When these ID blocks are in the dependency graph,
 * we can get rid of the manual dependency checks.
 * \{ */

static void material_changed(Main *UNUSED(bmain), Material *ma)
{
  /* icons */
  BKE_icon_changed(BKE_icon_id_ensure(&ma->id));
}

static void lamp_changed(Main *UNUSED(bmain), Light *la)
{
  /* icons */
  BKE_icon_changed(BKE_icon_id_ensure(&la->id));
}

static void texture_changed(Main *bmain, Tex *tex)
{
  Scene *scene;
  ViewLayer *view_layer;
  bNode *node;

  /* icons */
  BKE_icon_changed(BKE_icon_id_ensure(&tex->id));

  for (scene = bmain->scenes.first; scene; scene = scene->id.next) {
    /* paint overlays */
    for (view_layer = scene->view_layers.first; view_layer; view_layer = view_layer->next) {
      BKE_paint_invalidate_overlay_tex(scene, view_layer, tex);
    }
    /* find compositing nodes */
    if (scene->use_nodes && scene->nodetree) {
      for (node = scene->nodetree->nodes.first; node; node = node->next) {
        if (node->id == &tex->id) {
          ED_node_tag_update_id(&scene->id);
        }
      }
    }
  }
}

static void world_changed(Main *UNUSED(bmain), World *wo)
{
  /* icons */
  BKE_icon_changed(BKE_icon_id_ensure(&wo->id));
}

static void image_changed(Main *bmain, Image *ima)
{
  Tex *tex;

  /* icons */
  BKE_icon_changed(BKE_icon_id_ensure(&ima->id));

  /* textures */
  for (tex = bmain->textures.first; tex; tex = tex->id.next) {
    if (tex->type == TEX_IMAGE && tex->ima == ima) {
      texture_changed(bmain, tex);
    }
  }
}

static void scene_changed(Main *bmain, Scene *scene)
{
  Object *ob;

  /* glsl */
  for (ob = bmain->objects.first; ob; ob = ob->id.next) {
    if (ob->mode & OB_MODE_TEXTURE_PAINT) {
      BKE_texpaint_slots_refresh_object(scene, ob);
      ED_paint_proj_mesh_data_check(scene, ob, NULL, NULL, NULL, NULL);
    }
  }
}

void ED_render_id_flush_update(const DEGEditorUpdateContext *update_ctx, ID *id)
{
  /* this can be called from render or baking thread when a python script makes
   * changes, in that case we don't want to do any editor updates, and making
   * GPU changes is not possible because OpenGL only works in the main thread */
  if (!BLI_thread_is_main()) {
    return;
  }
  Main *bmain = update_ctx->bmain;
  /* Internal ID update handlers. */
  switch (GS(id->name)) {
    case ID_MA:
      material_changed(bmain, (Material *)id);
      break;
    case ID_TE:
      texture_changed(bmain, (Tex *)id);
      break;
    case ID_WO:
      world_changed(bmain, (World *)id);
      break;
    case ID_LA:
      lamp_changed(bmain, (Light *)id);
      break;
    case ID_IM:
      image_changed(bmain, (Image *)id);
      break;
    case ID_SCE:
      scene_changed(bmain, (Scene *)id);
      break;
    default:
      break;
  }
}

/** \} */
