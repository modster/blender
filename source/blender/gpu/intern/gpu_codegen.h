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
 * The Original Code is Copyright (C) 2005 Blender Foundation.
 * All rights reserved.
 */

/** \file
 * \ingroup gpu
 *
 * Generate shader code from the intermediate node graph.
 */

#pragma once

#include "GPU_material.h"
#include "GPU_shader.h"

#ifdef __cplusplus
extern "C" {
#endif

struct GPUNodeGraph;

typedef struct GPUPass {
  struct GPUPass *next;

  GPUShader *shader;
  GPUShaderSource source;
  uint refcount; /* Orphaned GPUPasses gets freed by the garbage collector. */
  uint32_t hash; /* Identity hash generated from all GLSL code. */
  bool compiled; /* Did we already tried to compile the attached GPUShader. */
} GPUPass;

/* Pass */

GPUPass *GPU_generate_pass(GPUMaterial *material,
                           struct GPUNodeGraph *graph,
                           GPUCodegenCallbackFn finalize_source_cb,
                           void *thunk);
GPUShader *GPU_pass_shader_get(GPUPass *pass);
bool GPU_pass_compile(GPUPass *pass, const char *shname);
void GPU_pass_release(GPUPass *pass);

/* Module */

void gpu_codegen_init(void);
void gpu_codegen_exit(void);

#ifdef __cplusplus
}
#endif
