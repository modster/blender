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
 * Copyright 2021, Blender Foundation.
 */

#include "BKE_global.h"

#include "GPU_framebuffer.h"

#include "ED_view3d.h"

#include "DRW_render.h"

#include "eevee_private.h"

#include "eevee_instance.hh"

using namespace blender::eevee;

static ShaderModule *g_shader_module = nullptr;

/* -------------------------------------------------------------------- */
/** \name EEVEE Instance C interface
 * \{ */

EEVEE_Instance *EEVEE_instance_alloc(void)
{
  if (g_shader_module == nullptr) {
    /* TODO(fclem) threadsafety. */
    g_shader_module = new ShaderModule();
  }
  return reinterpret_cast<EEVEE_Instance *>(new Instance(*g_shader_module));
}

void EEVEE_instance_free(EEVEE_Instance *instance_)
{
  Instance *instance = reinterpret_cast<Instance *>(instance_);
  delete instance;
}

void EEVEE_instance_init(EEVEE_Instance *instance_)
{
  Instance *instance = reinterpret_cast<Instance *>(instance_);

  const DRWContextState *ctx_state = DRW_context_state_get();
  Depsgraph *depsgraph = ctx_state->depsgraph;
  Scene *scene = ctx_state->scene;
  View3D *v3d = ctx_state->v3d;
  const ARegion *region = ctx_state->region;
  RegionView3D *rv3d = ctx_state->rv3d;

  /* Scaling output to better see what happens with accumulation. */
  int resolution_divider = (ELEM(G.debug_value, 1, 2)) ? 16 : 1;

  DefaultTextureList *dtxl = DRW_viewport_texture_list_get();
  int size[2];
  size[0] = divide_ceil_u(GPU_texture_width(dtxl->color), resolution_divider);
  size[1] = divide_ceil_u(GPU_texture_height(dtxl->color), resolution_divider);

  const DRWView *default_view = DRW_view_default_get();

  Object *camera = nullptr;
  /* Get render borders. */
  rcti rect;
  BLI_rcti_init(&rect, 0, size[0], 0, size[1]);
  if (v3d) {
    if (rv3d && (rv3d->persp == RV3D_CAMOB)) {
      camera = v3d->camera;
    }

    if (v3d->flag2 & V3D_RENDER_BORDER) {
      if (camera) {
        rctf viewborder;
        /* TODO(fclem) Might be better to get it from DRW. */
        ED_view3d_calc_camera_border(scene, depsgraph, region, v3d, rv3d, &viewborder, false);
        float viewborder_sizex = BLI_rctf_size_x(&viewborder);
        float viewborder_sizey = BLI_rctf_size_y(&viewborder);
        rect.xmin = floorf(viewborder.xmin + (scene->r.border.xmin * viewborder_sizex));
        rect.ymin = floorf(viewborder.ymin + (scene->r.border.ymin * viewborder_sizey));
        rect.xmax = floorf(viewborder.xmin + (scene->r.border.xmax * viewborder_sizex));
        rect.ymax = floorf(viewborder.ymin + (scene->r.border.ymax * viewborder_sizey));
      }
      else {
        rect.xmin = v3d->render_border.xmin * size[0];
        rect.ymin = v3d->render_border.ymin * size[1];
        rect.xmax = v3d->render_border.xmax * size[0];
        rect.ymax = v3d->render_border.ymax * size[1];
      }
    }
  }

  instance->init(size, &rect, nullptr, depsgraph, camera, nullptr, default_view, v3d, rv3d);
}

void EEVEE_instance_cache_init(EEVEE_Instance *instance_)
{
  Instance *instance = reinterpret_cast<Instance *>(instance_);
  instance->begin_sync();
}

void EEVEE_instance_cache_populate(EEVEE_Instance *instance_, Object *object)
{
  Instance *instance = reinterpret_cast<Instance *>(instance_);
  instance->object_sync(object);
}

void EEVEE_instance_cache_finish(EEVEE_Instance *instance_)
{
  Instance *instance = reinterpret_cast<Instance *>(instance_);
  instance->end_sync();
}

void EEVEE_instance_draw_viewport(EEVEE_Instance *instance_)
{
  Instance *instance = reinterpret_cast<Instance *>(instance_);
  DefaultFramebufferList *dfbl = DRW_viewport_framebuffer_list_get();

  instance->draw_viewport(dfbl);
}

void EEVEE_instance_render_frame(EEVEE_Instance *instance_,
                                 struct RenderEngine *engine,
                                 struct RenderLayer *render_layer)
{
  Instance *instance = reinterpret_cast<Instance *>(instance_);
  Render *render = engine->re;
  Depsgraph *depsgraph = DRW_context_state_get()->depsgraph;
  Object *camera_original_ob = RE_GetCamera(engine->re);
  const char *viewname = RE_GetActiveRenderView(engine->re);
  int size[2] = {engine->resolution_x, engine->resolution_y};

  rctf view_rect;
  rcti rect;
  RE_GetViewPlane(render, &view_rect, &rect);

  instance->init(size, &rect, engine, depsgraph, camera_original_ob, render_layer);
  instance->render_frame(render_layer, viewname);
}

void EEVEE_instance_view_update(EEVEE_Instance *instance_)
{
  Instance *instance = reinterpret_cast<Instance *>(instance_);
  instance->view_update();
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name EEVEE Shaders C interface
 * \{ */

void EEVEE_shared_data_free(void)
{
  delete g_shader_module;
  g_shader_module = nullptr;
}

/** \} */
