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

  /* Scaling output to better see what happens with accumulation. */
  int resolution_divider = (ELEM(G.debug_value, 1, 2)) ? 16 : 1;

  DefaultTextureList *dtxl = DRW_viewport_texture_list_get();
  int size[2];
  size[0] = divide_ceil_u(GPU_texture_width(dtxl->color), resolution_divider);
  size[1] = divide_ceil_u(GPU_texture_height(dtxl->color), resolution_divider);

  const DRWView *default_view = DRW_view_default_get();

  Object *camera = NULL;

  if (ctx_state->v3d && ctx_state->rv3d && (ctx_state->rv3d->persp == RV3D_CAMOB)) {
    camera = ctx_state->v3d->camera;
  }

  instance->init(size,
                 ctx_state->scene,
                 ctx_state->view_layer,
                 ctx_state->depsgraph,
                 camera,
                 default_view,
                 ctx_state->rv3d);
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

  instance->render_sample();

  instance->render_passes.combined->resolve_onto(dfbl->color_only_fb);
  instance->render_passes.depth->resolve_onto(dfbl->depth_only_fb);

  // if (instance->lookdev) {
  // instance->lookdev->resolve_onto(dfbl->default_fb);
  // }

  if (!instance->finished()) {
    DRW_viewport_request_redraw();
  }
}

#if 0 /* TODO Rendering */
void EEVEE_instance_render_frame(EEVEE_Instance *instance)
{
  instance->init(ctx_state->depsgraph, nullptr, size);

  while (instance->render_sample()) {
    /* TODO(fclem) print progression. */
  }

  instance->passes.combined->read_to_memory(rr->);
}
#endif

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
