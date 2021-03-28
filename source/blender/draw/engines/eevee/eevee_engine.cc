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

#include "GPU_framebuffer.h"

#include "DRW_render.h"

#include "eevee_private.h"

#include "eevee_instance.hh"

static EEVEE_Shaders *g_shaders = nullptr;

/* -------------------------------------------------------------------- */
/** \name EEVEE Instance C interface
 * \{ */

EEVEE_Instance *EEVEE_instance_alloc(void)
{
  if (g_shaders == nullptr) {
    /* TODO(fclem) threadsafety. */
    g_shaders = new EEVEE_Shaders();
  }
  return new EEVEE_Instance(*g_shaders);
}

void EEVEE_instance_free(EEVEE_Instance *instance)
{
  delete instance;
}

void EEVEE_instance_init(EEVEE_Instance *instance)
{
  const DRWContextState *ctx_state = DRW_context_state_get();

  DefaultTextureList *dtxl = DRW_viewport_texture_list_get();
  int size[2];
  size[0] = GPU_texture_width(dtxl->color);
  size[1] = GPU_texture_height(dtxl->color);

  const DRWView *default_view = DRW_view_default_get();

  instance->init(
      size, ctx_state->scene, ctx_state->view_layer, ctx_state->depsgraph, nullptr, default_view);
}

void EEVEE_instance_cache_init(EEVEE_Instance *instance)
{
  instance->begin_sync();
}

void EEVEE_instance_cache_populate(EEVEE_Instance *instance, Object *object)
{
  instance->object_sync(object);
}

void EEVEE_instance_cache_finish(EEVEE_Instance *instance)
{
  instance->end_sync();
}

void EEVEE_instance_draw_viewport(EEVEE_Instance *instance)
{
  DefaultFramebufferList *dfbl = DRW_viewport_framebuffer_list_get();

  instance->render_sample();

  instance->render_passes.combined->resolve_onto(dfbl->default_fb);

  // if (instance->lookdev) {
  // instance->lookdev->resolve_onto(dfbl->default_fb);
  // }
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
  delete g_shaders;
  g_shaders = nullptr;
}

/** \} */
