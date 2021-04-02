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

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

struct Object;
struct EEVEE_Instance;

struct EEVEE_Instance *EEVEE_instance_alloc(void);
void EEVEE_instance_free(struct EEVEE_Instance *instance_data_);

void EEVEE_instance_init(struct EEVEE_Instance *instance);

void EEVEE_instance_cache_init(struct EEVEE_Instance *instance);
void EEVEE_instance_cache_populate(struct EEVEE_Instance *instance, struct Object *object);
void EEVEE_instance_cache_finish(struct EEVEE_Instance *instance);

void EEVEE_instance_draw_viewport(struct EEVEE_Instance *instance_data_);
void EEVEE_instance_render_frame(struct EEVEE_Instance *instance_data_,
                                 struct RenderEngine *engine,
                                 struct RenderLayer *layer);

void EEVEE_shared_data_free(void);

#ifdef __cplusplus
}
#endif
