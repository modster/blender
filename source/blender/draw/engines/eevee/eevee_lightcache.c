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
 *
 * Eevee's indirect lighting cache.
 */

#include "DRW_render.h"

#include "BKE_global.h"

#include "BLI_endian_switch.h"
#include "BLI_threads.h"

#include "DEG_depsgraph_build.h"
#include "DEG_depsgraph_query.h"

#include "BKE_object.h"

#include "DNA_collection_types.h"
#include "DNA_lightprobe_types.h"

#include "PIL_time.h"

#include "eevee_lightcache.h"

#include "GPU_capabilities.h"
#include "GPU_context.h"

#include "WM_api.h"
#include "WM_types.h"

#include "BLO_read_write.h"

#include "wm_window.h"

/* TODO should be replace by a more elegant alternative. */
extern void DRW_opengl_context_enable(void);
extern void DRW_opengl_context_disable(void);

extern void DRW_opengl_render_context_enable(void *re_gl_context);
extern void DRW_opengl_render_context_disable(void *re_gl_context);
extern void DRW_gpu_render_context_enable(void *re_gpu_context);
extern void DRW_gpu_render_context_disable(void *re_gpu_context);

typedef struct EEVEE_LightBake {
} EEVEE_LightBake;

/* -------------------------------------------------------------------- */
/** \name Light Cache
 * \{ */

/* Return memory footprint in bytes. */
static uint eevee_lightcache_memsize_get(LightCache *lcache)
{
  return 0;
}

static bool eevee_lightcache_version_check(const LightCache *lcache)
{
  return false;
}

static bool eevee_lightcache_can_be_saved(LightCache *lcache)
{
  return true;
}

static int eevee_lightcache_irradiance_sample_count(LightCache *lcache)
{
  return 0;
}

void EEVEE_lightcache_info_update(SceneEEVEE *eevee)
{
}

static void irradiance_pool_size_get(int visibility_size, int total_samples, int r_size[3])
{
}

static bool EEVEE_lightcache_validate(const LightCache *light_cache,
                                      const int cube_len,
                                      const int cube_res,
                                      const int grid_len,
                                      const int irr_size[3])
{
  return false;
}

LightCache *EEVEE_lightcache_create(const int grid_len,
                                    const int cube_len,
                                    const int cube_size,
                                    const int vis_size,
                                    const int irr_size[3])
{
  return NULL;
}

static bool eevee_lightcache_static_load(LightCache *lcache)
{
  return false;
}

bool EEVEE_lightcache_load(LightCache *lcache)
{
  return false;
}

static void eevee_lightbake_readback_irradiance(LightCache *lcache)
{
}

static void eevee_lightbake_readback_reflections(LightCache *lcache)
{
}

void EEVEE_lightcache_free(LightCache *lcache)
{
}

static void write_lightcache_texture(BlendWriter *writer, LightCacheTexture *tex)
{
}

void EEVEE_lightcache_blend_write(BlendWriter *writer, LightCache *cache)
{
}

static void direct_link_lightcache_texture(BlendDataReader *reader, LightCacheTexture *lctex)
{
}

void EEVEE_lightcache_blend_read_data(BlendDataReader *reader, LightCache *cache)
{
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Light Bake Context
 * \{ */

static void eevee_lightbake_context_enable(EEVEE_LightBake *lbake)
{
}

static void eevee_lightbake_context_disable(EEVEE_LightBake *lbake)
{
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Light Bake Job
 * \{ */

wmJob *EEVEE_lightbake_job_create(struct wmWindowManager *wm,
                                  struct wmWindow *win,
                                  struct Main *bmain,
                                  struct ViewLayer *view_layer,
                                  struct Scene *scene,
                                  int delay,
                                  int frame)
{
  return NULL;
}

/* MUST run on the main thread. */
void *EEVEE_lightbake_job_data_alloc(struct Main *bmain,
                                     struct ViewLayer *view_layer,
                                     struct Scene *scene,
                                     bool run_as_job,
                                     int frame)
{
  return NULL;
}

void EEVEE_lightbake_job_data_free(void *custom_data)
{
}

void EEVEE_lightbake_update(void *custom_data)
{
}

void EEVEE_lightbake_job(void *custom_data, short *stop, short *do_update, float *progress)
{
}

/** \} */
