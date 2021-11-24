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
 * Descriptior type used to define shader structure, resources and interfaces.
 *
 * Some rule of thumb:
 * - Do not include anything else than this file in each descriptor file.
 * - You can assume all descriptors will be defined and you can reference any of them in
 *   additional_descriptors.
 */

#pragma once

#include <stdbool.h>

/* For helping code suggestion. */
#ifndef GPU_SHADER_DESCRIPTOR
#  define GPU_STAGE_INTERFACE_CREATE(_interface, ...) GPUInOut _interface[] = __VA_ARGS__;
#  define GPU_SHADER_DESCRIPTOR(_descriptor, ...) GPUShaderDescriptor _descriptor = __VA_ARGS__;
#endif

#ifndef ARRAY_SIZE
#  define ARRAY_SIZE(arr) (sizeof(arr) / sizeof(*(arr)))
#endif

#define UNUSED 0

typedef enum eGPUInOutType {
  /* UNUSED = 0 */
  FLOAT = 1,
  VEC2,
  VEC3,
  VEC4,
  UINT,
  UVEC2,
  UVEC3,
  UVEC4,
  INT,
  IVEC2,
  IVEC3,
  IVEC4,
  BOOL,
  /* Samplers & images. */
  FLOAT_BUFFER,
  FLOAT_1D,
  FLOAT_1D_ARRAY,
  FLOAT_2D,
  FLOAT_2D_ARRAY,
  FLOAT_3D,
  INT_BUFFER,
  INT_1D,
  INT_1D_ARRAY,
  INT_2D,
  INT_2D_ARRAY,
  INT_3D,
  UINT_BUFFER,
  UINT_1D,
  UINT_1D_ARRAY,
  UINT_2D,
  UINT_2D_ARRAY,
  UINT_3D,
  /* Custom structure. */
  STRUCT,
} eGPUInOutType;

typedef enum eGPUInOutQualifier {
  /* Storage qualifiers. */
  RESTRICT = (1 << 0),
  READ_ONLY = (1 << 1),
  WRITE_ONLY = (1 << 2),
  /* Interp qualifiers. */
  SMOOTH = (1 << 3),
  FLAT = (2 << 3),
  NO_PERSPECTIVE = (3 << 3),
  /* Dual Source Blending Index. */
  OUTPUT_INDEX_0 = (4 << 3),
  OUTPUT_INDEX_1 = (5 << 3),
} eGPUInOutQualifier;

typedef enum eGPUBindType {
  GPU_BIND_UNIFORM_BUFFER = 1,
  GPU_BIND_STORAGE_BUFFER,
  GPU_BIND_SAMPLER,
  GPU_BIND_IMAGE,
} eGPUBindType;

/* Syntaxic suggar. */
#define DESCRIPTOR_SET_0 0
#define DESCRIPTOR_SET_1 1
#define DESCRIPTOR_SET_2 2
#define DESCRIPTOR_SET_3 3

typedef struct GPUInOut {
  eGPUInOutType type;
  const char *name;
  eGPUInOutQualifier qual;
} GPUInOut;

#define VERTEX_INPUT(_type, _name) \
  { \
    .type = _type, .name = _name \
  }

#define FRAGMENT_OUTPUT(_type, _name) \
  { \
    .type = _type, .name = _name \
  }

#define FRAGMENT_OUTPUT_DUALBLEND(_type, _name, _index) \
  { \
    .type = _type, .name = _name, .qual = _index \
  }

#define PUSH_CONSTANT(_type, _name) \
  { \
    .type = _type, .name = _name \
  }

typedef enum eGPUSamplerState {
  GPU_SAMPLER_DEFAULT = 0,
  GPU_SAMPLER_FILTER = (1 << 0),
  GPU_SAMPLER_MIPMAP = (1 << 1),
  GPU_SAMPLER_REPEAT_S = (1 << 2),
  GPU_SAMPLER_REPEAT_T = (1 << 3),
  GPU_SAMPLER_REPEAT_R = (1 << 4),
  GPU_SAMPLER_CLAMP_BORDER = (1 << 5), /* Clamp to border color instead of border texel. */
  GPU_SAMPLER_COMPARE = (1 << 6),
  GPU_SAMPLER_ANISO = (1 << 7),
  GPU_SAMPLER_ICON = (1 << 8),

  GPU_SAMPLER_REPEAT = (GPU_SAMPLER_REPEAT_S | GPU_SAMPLER_REPEAT_T | GPU_SAMPLER_REPEAT_R),
} eGPUSamplerState;

typedef struct GPUResourceBind {
  eGPUBindType bind_type;
  eGPUInOutType type;
  eGPUInOutQualifier qual;
  eGPUSamplerState sampler;
  /* Defined type name. */
  const char *type_name;
  /* Name can contain array size (i.e: "colors[6]"). Note: supports unsized arrays for SSBO. */
  const char *name;
} GPUResourceBind;

#define UNIFORM_BUFFER(_typename, _name) \
  { \
    .bind_type = GPU_BIND_UNIFORM_BUFFER, .type_name = _typename, .name = _name, \
  }

#define STORAGE_BUFFER(_typename, _name, _qual) \
  { \
    .bind_type = GPU_BIND_STORAGE_BUFFER, .type_name = _typename, .name = _name, .qual = _qual, \
  }

#define SAMPLER(_type, _name, _sampler) \
  { \
    .bind_type = GPU_BIND_SAMPLER, .type = _type, .name = _name, .sampler = _sampler, \
  }

#define IMAGE(_type, _name, _qual) \
  { \
    .bind_type = GPU_BIND_IMAGE, .type = _type, .name = _name, .qual = _qual, \
  }

typedef struct GPUInterfaceBlockDescription {
  /** Name of the instance of the block (used to access). Can be empty "". */
  const char *name;
  /** List of all members of the interface. */
  int inouts_len;
  GPUInOut *inouts;
} GPUInterfaceBlockDescription;

#define STAGE_INTERFACE(_name, _inouts) \
  { \
    .name = _name, .inouts_len = ARRAY_SIZE(_inouts), .inouts = _inouts \
  }

/* Vulkan garantee 4 distinct descriptior set. */
#define GPU_MAX_DESCIPTOR_SET 2
/* Should be tweaked to be as large as the maximum supported by the low end hardware we support. */
#define GPU_MAX_RESOURCE_PER_DESCRIPTOR 8

/**
 * @brief Describe inputs & outputs, stage interfaces, resources and sources of a shader.
 *        If all data is correctly provided, this is all that is needed to create and compile
 *        a GPUShader.
 *
 * IMPORTANT: All strings are references only. Make sure all the strings used by a
 *            GPUShaderDescriptor are not freed until it is consumed or deleted.
 */
typedef struct GPUShaderDescriptor {
  /** Shader name for debugging. */
  const char *name;
  /** True if the shader is static and can be precompiled at compile time. */
  bool do_static_compilation;

  GPUInOut vertex_inputs[16];
  GPUInOut fragment_outputs[8];

  GPUInterfaceBlockDescription vertex_out_interfaces[4];
  GPUInterfaceBlockDescription geometry_out_interfaces[4];

  GPUResourceBind resources[GPU_MAX_DESCIPTOR_SET][GPU_MAX_RESOURCE_PER_DESCRIPTOR];

  /**
   * Data managed by GPUShader. Can be set through uniform functions. Must be less than 128bytes.
   * One slot represents 4bytes. Each element needs to have enough empty space left after it.
   * example:
   * [0] = PUSH_CONSTANT(MAT4, "ModelMatrix"),
   * ---- 16 slots occupied by ModelMatrix ----
   * [16] = PUSH_CONSTANT(VEC4, "color"),
   * ---- 4 slots occupied by color ----
   * [20] = PUSH_CONSTANT(BOOL, "srgbToggle"),
   */
  GPUInOut push_constants[32];

  const char *vertex_source, *geometry_source, *fragment_source, *compute_source;

  const char *defines[8];

  /**
   * Link to other descriptors to recursively merge with this one.
   * No data slot must overlap otherwise we throw an error.
   */
  struct GPUShaderDescriptor *additional_descriptors[4];
} GPUShaderDescriptor;
