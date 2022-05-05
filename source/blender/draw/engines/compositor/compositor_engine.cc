/* SPDX-License-Identifier: GPL-2.0-or-later */

#include "BLI_listbase.h"
#include "BLI_math_vec_types.hh"
#include "BLI_string_ref.hh"

#include "BLT_translation.h"

#include "DNA_ID_enums.h"
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

class DRWContext : public Context {
 public:
  using Context::Context;

  const Scene *get_scene() const override
  {
    return DRW_context_state_get()->scene;
  }

  int2 get_viewport_size() override
  {
    return int2(float2(DRW_viewport_size_get()));
  }

  GPUTexture *get_viewport_texture() override
  {
    return DRW_viewport_texture_list_get()->color;
  }

  GPUTexture *get_pass_texture(int view_layer, eScenePassType pass_type) override
  {
    return get_viewport_texture();
  }

  StringRef get_view_name() override
  {
    const SceneRenderView *view = static_cast<SceneRenderView *>(
        BLI_findlink(&get_scene()->r.views, DRW_context_state_get()->v3d->multiview_eye));
    return view->name;
  }
};

class Engine {
 private:
  DRWTexturePool texture_pool_;
  DRWContext context_;
  Evaluator evaluator_;
  /* Stores the viewport size at the time the last compositor evaluation happened. See the
   * update_viewport_size method for more information. */
  int2 viewport_size_;

 public:
  Engine()
      : context_(texture_pool_),
        evaluator_(context_, node_tree()),
        viewport_size_(context_.get_viewport_size())
  {
  }

  /* Update the viewport size and evaluate the compositor. */
  void draw()
  {
    update_viewport_size();
    evaluator_.evaluate();
  }

  /* If the size of the viewport changed from the last time the compositor was evaluated, update
   * the viewport size and reset the evaluator. That's because the evaluator compiles the node tree
   * in a manner that is specifically optimized for the size of the viewport. This should be called
   * before evaluating the compositor. */
  void update_viewport_size()
  {
    if (viewport_size_ == context_.get_viewport_size()) {
      return;
    }

    viewport_size_ = context_.get_viewport_size();

    evaluator_.reset();
  }

  /* If the compositor node tree changed, reset the evaluator. */
  void update(const Depsgraph *depsgraph)
  {
    if (DEG_id_type_updated(depsgraph, ID_NT)) {
      evaluator_.reset();
    }
  }

  /* Get a reference to the compositor node tree. */
  static bNodeTree &node_tree()
  {
    return *DRW_context_state_get()->scene->nodetree;
  }
};

}  // namespace blender::viewport_compositor

using namespace blender::viewport_compositor;

typedef struct CompositorData {
  DrawEngineType *engine_type;
  DRWViewportEmptyList *fbl;
  DRWViewportEmptyList *txl;
  DRWViewportEmptyList *psl;
  DRWViewportEmptyList *stl;
  Engine *instance_data;
} CompositorData;

static void compositor_engine_init(void *data)
{
  CompositorData *compositor_data = static_cast<CompositorData *>(data);

  if (!compositor_data->instance_data) {
    compositor_data->instance_data = new Engine();
  }
}

static void compositor_engine_free(void *instance_data)
{
  Engine *engine = static_cast<Engine *>(instance_data);
  delete engine;
}

static void compositor_engine_draw(void *data)
{
  const CompositorData *compositor_data = static_cast<CompositorData *>(data);
  compositor_data->instance_data->draw();
}

static void compositor_engine_update(void *data)
{
  const CompositorData *compositor_data = static_cast<CompositorData *>(data);
  compositor_data->instance_data->update(DRW_context_state_get()->depsgraph);
}

extern "C" {

static const DrawEngineDataSize compositor_data_size = DRW_VIEWPORT_DATA_SIZE(CompositorData);

DrawEngineType draw_engine_compositor_type = {
    nullptr,                   /* next */
    nullptr,                   /* prev */
    N_("Compositor"),          /* idname */
    &compositor_data_size,     /* vedata_size */
    &compositor_engine_init,   /* engine_init */
    nullptr,                   /* engine_free */
    &compositor_engine_free,   /* instance_free */
    nullptr,                   /* cache_init */
    nullptr,                   /* cache_populate */
    nullptr,                   /* cache_finish */
    &compositor_engine_draw,   /* draw_scene */
    &compositor_engine_update, /* view_update */
    nullptr,                   /* id_update */
    nullptr,                   /* render_to_image */
    nullptr,                   /* store_metadata */
};
}
