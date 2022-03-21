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
 * Parsing of and code generation using GLSL shaders in gpu/shaders/material. */

#pragma once

#include "GPU_material.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_FUNCTION_NAME 64
#define MAX_PARAMETER 36

struct GSet;

typedef enum {
  FUNCTION_QUAL_NONE = 0,
  FUNCTION_QUAL_IN = 1 << 0,
  FUNCTION_QUAL_OUT = 1 << 1,
  FUNCTION_QUAL_INOUT = 1 << 2,
  FUNCTION_QUAL_CONST = 1 << 3,
  FUNCTION_QUAL_RESTRICT = 1 << 4,
  FUNCTION_QUAL_WRITEONLY = 1 << 5,
} GPUFunctionQual;

typedef struct GPUFunction {
  char name[MAX_FUNCTION_NAME];
  eGPUType paramtype[MAX_PARAMETER];
  GPUFunctionQual paramqual[MAX_PARAMETER];
  int totparam;
  /* TOOD(@fclem): Clean that void pointer. */
  void *source; /* GPUSource */
} GPUFunction;

GPUFunction *gpu_material_library_use_function(struct GSet *used_libraries, const char *name);

#ifdef __cplusplus
}
#endif
