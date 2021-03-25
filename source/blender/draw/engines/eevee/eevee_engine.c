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
}

static void eevee_draw_scene(void *vedata)
{
  EEVEE_instance_draw_viewport(((EEVEE_Data *)vedata)->instance_data);
}

static void eevee_engine_free(void)
{
  /* Free all static resources. */
}

static void eevee_instance_free(void *instance_data)
{
  EEVEE_instance_free((struct EEVEE_Instance *)instance_data);
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
    NULL,
    NULL,
    NULL,
    &eevee_draw_scene,
    NULL,
    NULL,
    NULL,
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
    &draw_engine_eevee_type,
    {NULL, NULL, NULL},
};

#undef EEVEE_ENGINE
