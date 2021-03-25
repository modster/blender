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

#include "eevee_private.h"

#include "eevee.hh"

EEVEE_Instance::EEVEE_Instance(void)
{
}

EEVEE_Instance::~EEVEE_Instance(void)
{
}

/* -------------------------------------------------------------------- */
/** \name C interface
 * \{ */

EEVEE_Instance *EEVEE_instance_alloc(void)
{
  return new EEVEE_Instance();
}

void EEVEE_instance_free(EEVEE_Instance *instance)
{
  delete instance;
}

void EEVEE_instance_draw_viewport(EEVEE_Instance *UNUSED(instance))
{
  float color[4] = {1, 0, 0, 1};

  GPUFrameBuffer *fb = GPU_framebuffer_active_get();
  GPU_framebuffer_clear_color(fb, color);
}

/** \} */
