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
 * Copyright 2018, Blender Foundation.
 */

/** \file
 * \ingroup eevee
 *
 * Contains functions used outside of EEVEE for lightcache baking.
 */

#pragma once

#include "BLI_sys_types.h" /* for bool */

#ifdef __cplusplus
extern "C" {
#endif

struct BlendDataReader;
struct BlendWriter;
struct EEVEE_Data;
struct EEVEE_ViewLayerData;
struct LightCache;
struct Scene;
struct SceneEEVEE;
struct ViewLayer;

/* Light Bake */
struct wmJob *EEVEE_lightbake_job_create(struct wmWindowManager *wm,
                                         struct wmWindow *win,
                                         struct Main *bmain,
                                         struct ViewLayer *view_layer,
                                         struct Scene *scene,
                                         int delay,
                                         int frame);
void *EEVEE_lightbake_job_data_alloc(struct Main *bmain,
                                     struct ViewLayer *view_layer,
                                     struct Scene *scene,
                                     bool run_as_job,
                                     int frame);
void EEVEE_lightbake_job_data_free(void *custom_data);
void EEVEE_lightbake_update(void *custom_data);
void EEVEE_lightbake_job(void *custom_data, short *stop, short *do_update, float *progress);

/* Light Cache */
void EEVEE_lightcache_free(struct LightCache *lcache);
void EEVEE_lightcache_info_update(struct SceneEEVEE *eevee);

void EEVEE_lightcache_blend_write(struct BlendWriter *writer, struct LightCache *cache);
void EEVEE_lightcache_blend_read_data(struct BlendDataReader *reader, struct LightCache *cache);

#ifdef __cplusplus
}
#endif
