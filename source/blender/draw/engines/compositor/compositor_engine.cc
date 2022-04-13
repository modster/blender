/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

#include "BLI_math_vec_types.hh"

#include "BLT_translation.h"

#include "DNA_scene_types.h"

#include "DEG_depsgraph_query.h"

#include "DRW_render.h"

#include "IMB_colormanagement.h"

#include "VPC_context.hh"
#include "VPC_evaluator.hh"
#include "VPC_texture_pool.hh"

#include "GPU_texture.h"

namespace blender::viewport_compositor {

class DRWTexturePool : public TexturePool {
 public:
  GPUTexture *allocate_texture(int2 size, eGPUTextureFormat format) override
  {
    DrawEngineType *owner = (DrawEngineType *)this;
    return DRW_texture_pool_query_2d(size.x, size.y, format, owner);
  }
};

static const Scene *get_context_scene()
{
  return DRW_context_state_get()->scene;
}

class DRWContext : public Context {
 public:
  using Context::Context;

  const Scene *get_scene() override
  {
    return get_context_scene();
  }

  GPUTexture *get_viewport_texture() override
  {
    return DRW_viewport_texture_list_get()->color;
  }

  GPUTexture *get_pass_texture(int view_layer, eScenePassType pass_type) override
  {
    Scene *original_scene = (Scene *)DEG_get_original_id(&DRW_context_state_get()->scene->id);
    return DRW_render_pass_find(original_scene, view_layer, pass_type)->pass_tx;
  }
};

/* It is sufficient to check for the scene node tree because the engine will not be enabled when
 * the viewport shading option is disabled. */
static bool is_compositor_enabled()
{
  const Scene *scene = get_context_scene();
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
  const Scene *scene = get_context_scene();
  Evaluator evaluator(context, *scene->nodetree);
  evaluator.evaluate();
}

}  // namespace blender::viewport_compositor

static void compositor_draw(void *UNUSED(data))
{
  blender::viewport_compositor::draw();
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
