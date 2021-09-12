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

/** \file
 * \ingroup draw_engine
 *
 * Engine processing the render buffer using GLSL to apply the scene compositing node tree.
 */

#include "DRW_render.h"

#include "compositor_shader.hh"

namespace blender::compositor {

static ShaderModule *g_shader_module = nullptr;

class Instance {
 private:
  ShaderModule &shaders;

  Depsgraph *depsgraph;
  Scene *scene;

  bool enabled;

 public:
  Instance(ShaderModule &shader_module) : shaders(shader_module)
  {
  }

  void init()
  {
    const DRWContextState *ctx_state = DRW_context_state_get();
    depsgraph = ctx_state->depsgraph;
    scene = ctx_state->scene;
    enabled = scene->use_nodes && scene->nodetree;

    if (!enabled) {
      return;
    }
  }

  void sync()
  {
    if (!enabled) {
      return;
    }
  }

  void draw()
  {
    if (!enabled) {
      return;
    }

    DefaultFramebufferList *dfbl = DRW_viewport_framebuffer_list_get();

    float col[4] = {1, 0, 0, 1};
    GPU_framebuffer_clear_color(dfbl->default_fb, col);

    /* If compositor did not end on the input buffer, swap the buffer and the input buffer */
  }
};

}  // namespace blender::compositor

/* -------------------------------------------------------------------- */
/** \name C interface
 * \{ */

using namespace blender::compositor;

typedef struct COMPOSITOR_Data {
  DrawEngineType *engine_type;
  DRWViewportEmptyList *fbl;
  DRWViewportEmptyList *txl;
  DRWViewportEmptyList *psl;
  DRWViewportEmptyList *stl;
  Instance *instance_data;
} COMPOSITOR_Data;

static void compositor_engine_init(void *vedata)
{
  COMPOSITOR_Data *ved = (COMPOSITOR_Data *)vedata;

  if (g_shader_module == nullptr) {
    /* TODO(fclem) threadsafety. */
    g_shader_module = new ShaderModule();
  }

  if (ved->instance_data == nullptr) {
    ved->instance_data = new Instance(*g_shader_module);
  }

  ved->instance_data->init();
}

static void compositor_engine_free(void)
{
  delete g_shader_module;
  g_shader_module = nullptr;
}

static void compositor_instance_free(void *instance_data_)
{
  Instance *instance_data = reinterpret_cast<Instance *>(instance_data_);
  delete instance_data;
}

static void compositor_cache_init(void *vedata)
{
  COMPOSITOR_Data *ved = (COMPOSITOR_Data *)vedata;
  ved->instance_data->sync();
}

static void compositor_draw(void *vedata)
{
  COMPOSITOR_Data *ved = (COMPOSITOR_Data *)vedata;
  ved->instance_data->draw();
}

/** \} */

extern "C" {

static const DrawEngineDataSize compositor_data_size = DRW_VIEWPORT_DATA_SIZE(COMPOSITOR_Data);

DrawEngineType draw_engine_compositor_type = {
    nullptr,
    nullptr,
    N_("Compositor"),
    &compositor_data_size,
    &compositor_engine_init,
    &compositor_engine_free,
    &compositor_instance_free,
    &compositor_cache_init,
    nullptr,
    nullptr,
    &compositor_draw,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};
}
