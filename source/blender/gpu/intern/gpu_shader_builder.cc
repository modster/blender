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
 * The Original Code is Copyright (C) 2021 Blender Foundation.
 * All rights reserved.
 */

/** \file
 * \ingroup gpu
 *
 * Compile time automation of shader compilation and validation.
 */

#include <iostream>

#include "gpu_shader_create_info.hh"
#include "gpu_shader_create_info_private.hh"
#include "gpu_shader_dependency_private.h"

#include "CLG_log.h"
#include "GPU_context.h"
#include "GPU_init_exit.h"

#include "GHOST_C-api.h"

int main(int argc, char const *argv[])
{
  if (argc < 2) {
    printf("Usage: shader_builder <data_file_to>\n");
    exit(1);
  }

  (void)argv;

#if 0 /* Make it compile. Somehow... (dependency with GPU module is hard). */
  GHOST_GLSettings glSettings = {0};
  GHOST_SystemHandle ghost_system = GHOST_CreateSystem();
  GHOST_ContextHandle ghost_context = GHOST_CreateOpenGLContext(ghost_system, glSettings);
  GHOST_ActivateOpenGLContext(ghost_context);
  struct GPUContext *context = GPU_context_create(nullptr);
  GPU_init();

  gpu_shader_create_info_compile_all();

  GPU_exit();
  GPU_backend_exit();
  GPU_context_discard(context);
  GHOST_DisposeOpenGLContext(ghost_system, ghost_context);
  GHOST_DisposeSystem(ghost_system);
#endif

  return 0;
}
