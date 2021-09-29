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
 * Copyright 2016, Blender Foundation.
 */

/** \file
 * \ingroup draw_engine
 */

#include "DRW_render.h"

#include "DRW_engine.h"
#include "GPU_framebuffer.h"

#include "eevee_engine.h"
#include "eevee_private.h"

typedef struct EEVEE_Data {
  DrawEngineType *engine_type;
  DRWViewportEmptyList *fbl;
  DRWViewportEmptyList *txl;
  DRWViewportEmptyList *psl;
  DRWViewportEmptyList *stl;
  struct EEVEE_Instance *instance_data;
} EEVEE_Data;

static void eevee_engine_init(void *vedata)
{
  EEVEE_Data *ved = (EEVEE_Data *)vedata;

  if (ved->instance_data == NULL) {
    ved->instance_data = EEVEE_instance_alloc();
  }

  EEVEE_instance_init(ved->instance_data);
}

static void eevee_draw_scene(void *vedata)
{
  EEVEE_instance_draw_viewport(((EEVEE_Data *)vedata)->instance_data);
}

static void eevee_cache_init(void *vedata)
{
  EEVEE_instance_cache_init(((EEVEE_Data *)vedata)->instance_data);
}

static void eevee_cache_populate(void *vedata, Object *object)
{
  EEVEE_instance_cache_populate(((EEVEE_Data *)vedata)->instance_data, object);
}

static void eevee_cache_finish(void *vedata)
{
  EEVEE_instance_cache_finish(((EEVEE_Data *)vedata)->instance_data);
}

static void eevee_engine_free(void)
{
  EEVEE_shared_data_free();
}

static void eevee_instance_free(void *instance_data)
{
  EEVEE_instance_free((struct EEVEE_Instance *)instance_data);
}

static void eevee_render_to_image(void *UNUSED(vedata),
                                  struct RenderEngine *engine,
                                  struct RenderLayer *layer,
                                  const struct rcti *UNUSED(rect))
{
  struct EEVEE_Instance *instance = EEVEE_instance_alloc();
  EEVEE_instance_render_frame(instance, engine, layer);
  EEVEE_instance_free(instance);
}

static void eevee_render_update_passes(RenderEngine *engine, Scene *scene, ViewLayer *view_layer)
{
  RE_engine_register_pass(engine, scene, view_layer, RE_PASSNAME_COMBINED, 4, "RGBA", SOCK_RGBA);

#define CHECK_PASS_LEGACY(name, type, channels, chanid) \
  if (view_layer->passflag & (SCE_PASS_##name)) { \
    RE_engine_register_pass( \
        engine, scene, view_layer, RE_PASSNAME_##name, channels, chanid, type); \
  } \
  ((void)0)
#define CHECK_PASS_EEVEE(name, type, channels, chanid) \
  if (view_layer->eevee.render_passes & (EEVEE_RENDER_PASS_##name)) { \
    RE_engine_register_pass( \
        engine, scene, view_layer, RE_PASSNAME_##name, channels, chanid, type); \
  } \
  ((void)0)

  CHECK_PASS_LEGACY(Z, SOCK_FLOAT, 1, "Z");
  CHECK_PASS_LEGACY(MIST, SOCK_FLOAT, 1, "Z");
  CHECK_PASS_LEGACY(NORMAL, SOCK_VECTOR, 3, "XYZ");
  CHECK_PASS_LEGACY(VECTOR, SOCK_RGBA, 4, "RGBA");
  CHECK_PASS_LEGACY(DIFFUSE_DIRECT, SOCK_RGBA, 3, "RGB");
  CHECK_PASS_LEGACY(DIFFUSE_COLOR, SOCK_RGBA, 3, "RGB");
  CHECK_PASS_LEGACY(GLOSSY_DIRECT, SOCK_RGBA, 3, "RGB");
  CHECK_PASS_LEGACY(GLOSSY_COLOR, SOCK_RGBA, 3, "RGB");
  CHECK_PASS_EEVEE(VOLUME_LIGHT, SOCK_RGBA, 3, "RGB");
  CHECK_PASS_LEGACY(EMIT, SOCK_RGBA, 3, "RGB");
  CHECK_PASS_LEGACY(ENVIRONMENT, SOCK_RGBA, 3, "RGB");
  CHECK_PASS_LEGACY(SHADOW, SOCK_RGBA, 3, "RGB");
  CHECK_PASS_LEGACY(AO, SOCK_RGBA, 3, "RGB");
  CHECK_PASS_EEVEE(BLOOM, SOCK_RGBA, 3, "RGB");

  LISTBASE_FOREACH (ViewLayerAOV *, aov, &view_layer->aovs) {
    if ((aov->flag & AOV_CONFLICT) != 0) {
      continue;
    }
    switch (aov->type) {
      case AOV_TYPE_COLOR:
        RE_engine_register_pass(engine, scene, view_layer, aov->name, 4, "RGBA", SOCK_RGBA);
        break;
      case AOV_TYPE_VALUE:
        RE_engine_register_pass(engine, scene, view_layer, aov->name, 1, "X", SOCK_FLOAT);
        break;
      default:
        break;
    }
  }
  // EEVEE_cryptomatte_update_passes(engine, scene, view_layer);

#undef CHECK_PASS_LEGACY
#undef CHECK_PASS_EEVEE
}

static const DrawEngineDataSize eevee_data_size = DRW_VIEWPORT_DATA_SIZE(EEVEE_Data);

DrawEngineType draw_engine_eevee_type = {
    NULL,
    NULL,
    N_("Eevee"),
    &eevee_data_size,
    &eevee_engine_init,
    &eevee_engine_free,
    &eevee_instance_free,
    &eevee_cache_init,
    &eevee_cache_populate,
    &eevee_cache_finish,
    &eevee_draw_scene,
    NULL,
    NULL,
    &eevee_render_to_image,
    NULL,
};

#define EEVEE_ENGINE "BLENDER_EEVEE"

RenderEngineType DRW_engine_viewport_eevee_type = {
    NULL,
    NULL,
    EEVEE_ENGINE,
    N_("Eevee"),
    RE_INTERNAL | RE_USE_PREVIEW | RE_USE_STEREO_VIEWPORT | RE_USE_GPU_CONTEXT,
    NULL,
    &DRW_render_to_image,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    &eevee_render_update_passes,
    &draw_engine_eevee_type,
    {NULL, NULL, NULL},
};

#undef EEVEE_ENGINE
