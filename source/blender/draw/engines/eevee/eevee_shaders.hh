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

/** \file
 * \ingroup eevee
 *
 * Shader module that manage shader libraries, deferred compilation,
 * and static shader usage.
 */

#pragma once

#include "BLI_vector.hh"

#include "DRW_render.h"
#include "GPU_shader.h"

using namespace blender;

extern char datatoc_common_fullscreen_vert_glsl[];
extern char datatoc_common_hair_lib_glsl[];
extern char datatoc_common_math_geom_lib_glsl[];
extern char datatoc_common_math_lib_glsl[];
extern char datatoc_common_view_lib_glsl[];

extern char datatoc_accumulator_accumulate_frag_glsl[];
extern char datatoc_accumulator_resolve_frag_glsl[];
extern char datatoc_mesh_frag_glsl[];
extern char datatoc_mesh_lib_glsl[];
extern char datatoc_mesh_vert_glsl[];

/* Keep alphabetical order and clean prefix. */
enum eEEVEEShaderType {
  ACCUMULATOR_ACCUMULATE = 0,
  ACCUMULATOR_RESOLVE,
  MESH, /* TEST */

  MAX_SHADER_TYPE,
};

typedef struct EEVEE_Shaders {
 private:
  struct EEVEE_ShaderDescription {
    const char *name;
    const char *vertex_shader_code;
    const char *geometry_shader_code;
    const char *fragment_shader_code;
  };

  DRWShaderLibrary *shader_lib_ = nullptr;
  Vector<GPUShader *> shaders_;
  Vector<EEVEE_ShaderDescription> shader_descriptions_;

 public:
  EEVEE_Shaders()
  {
    shaders_.resize(MAX_SHADER_TYPE);
    shader_descriptions_.resize(MAX_SHADER_TYPE);

    for (GPUShader *&shader : shaders_) {
      shader = nullptr;
    }

    shader_lib_ = DRW_shader_library_create();
    /* NOTE: These need to be ordered by dependencies. */
    DRW_SHADER_LIB_ADD(shader_lib_, common_math_lib);
    DRW_SHADER_LIB_ADD(shader_lib_, common_math_geom_lib);
    DRW_SHADER_LIB_ADD(shader_lib_, common_hair_lib);
    DRW_SHADER_LIB_ADD(shader_lib_, common_view_lib);
    DRW_SHADER_LIB_ADD(shader_lib_, mesh_lib);

    /* Meh ¯\_(ツ)_/¯. */
    char *datatoc_nullptr_glsl = nullptr;

#define SHADER(enum_, vert_, geom_, frag_) \
  shader_descriptions_[enum_].name = STRINGIFY(enum_); \
  shader_descriptions_[enum_].vertex_shader_code = datatoc_##vert_##_glsl; \
  shader_descriptions_[enum_].geometry_shader_code = datatoc_##geom_##_glsl; \
  shader_descriptions_[enum_].fragment_shader_code = datatoc_##frag_##_glsl;

#define SHADER_FULLSCREEN(enum_, frag_) SHADER(enum_, common_fullscreen_vert, nullptr, frag_)

    SHADER_FULLSCREEN(ACCUMULATOR_ACCUMULATE, accumulator_accumulate_frag);
    SHADER_FULLSCREEN(ACCUMULATOR_RESOLVE, accumulator_resolve_frag);
    SHADER(MESH, mesh_vert, nullptr, mesh_frag);

#undef SHADER
#undef SHADER_FULLSCREEN

#ifdef DEBUG
    /* Ensure all shader are described. */
    for (EEVEE_ShaderDescription &desc : shader_descriptions_) {
      BLI_assert(desc.name != nullptr);
      BLI_assert(desc.vertex_shader_code != nullptr);
      BLI_assert(desc.fragment_shader_code != nullptr);
    }
#endif
  }

  ~EEVEE_Shaders()
  {
    for (GPUShader *&shader : shaders_) {
      DRW_SHADER_FREE_SAFE(shader);
    }
    DRW_SHADER_LIB_FREE_SAFE(shader_lib_);
  }

  GPUShader *static_shader_get(eEEVEEShaderType shader_type)
  {
    if (shaders_[shader_type] == nullptr) {
      EEVEE_ShaderDescription &desc = shader_descriptions_[shader_type];
      shaders_[shader_type] = DRW_shader_create_with_shaderlib_ex(desc.vertex_shader_code,
                                                                  desc.geometry_shader_code,
                                                                  desc.fragment_shader_code,
                                                                  shader_lib_,
                                                                  nullptr,
                                                                  desc.name);
      if (shaders_[shader_type] == nullptr) {
        fprintf(stderr, "EEVEE: error: Could not compile static shader \"%s\".", desc.name);
      }
      BLI_assert(shaders_[shader_type] != nullptr);
    }
    return shaders_[shader_type];
  }
} EEVEE_Shaders;