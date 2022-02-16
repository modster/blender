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

#include "DNA_scene_types.h"

#include "DRW_render.h"

#include "IMB_colormanagement.h"

#include "NOD_compositor_execute.hh"

#include "GPU_texture.h"

#include "compositor_shader.hh"

namespace blender::viewport_compositor {

class DRWTexturePool : public TexturePool {
 public:
  GPUTexture *allocate_texture(int width, int height, eGPUTextureFormat format) override
  {
    DrawEngineType *owner = (DrawEngineType *)this;
    return DRW_texture_pool_query_2d(width, height, format, owner);
  }
};

class DRWContext : public Context {
 public:
  using Context::Context;
  GPUTexture *get_viewport_texture() override
  {
    return DRW_viewport_texture_list_get()->color;
  }

  GPUTexture *get_pass_texture(int view_layer, eScenePassType pass_type) override
  {
    return DRW_render_pass_find(DRW_context_state_get()->scene, view_layer, pass_type)->pass_tx;
  }
};

/* Get the scene which includes the compositor node tree. */
static const Scene *get_scene()
{
  return DRW_context_state_get()->scene;
}

/* It is sufficient to check for the scene node tree because the engine will not be enabled when
 * the viewport shading option is disabled. */
static bool is_compositor_enabled()
{
  const Scene *scene = get_scene();
  if (scene->use_nodes && scene->nodetree) {
    return true;
  }
  return false;
}

static void draw()
{
  if (!is_compositor_enabled()) {
    return;
  }

  /* Reset default view. */
  DRW_view_set_active(nullptr);

  DRWTexturePool texture_pool;
  DRWContext context(texture_pool);
  const Scene *scene = get_scene();
  Evaluator evaluator(context, scene->nodetree);
  evaluator.evaluate();
}

}  // namespace blender::viewport_compositor

using namespace blender::viewport_compositor;

static void compositor_draw(void *UNUSED(data))
{
  draw();
}

extern "C" {

static const DrawEngineDataSize compositor_data_size = {};

DrawEngineType draw_engine_compositor_type = {
    nullptr,
    nullptr,
    N_("Compositor"),
    &compositor_data_size,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    &compositor_draw,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};
}
