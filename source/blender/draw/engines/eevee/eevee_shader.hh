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

#include <array>

#include "DRW_render.h"
#include "GPU_shader.h"

extern char datatoc_common_fullscreen_vert_glsl[];
extern char datatoc_common_hair_lib_glsl[];
extern char datatoc_common_math_geom_lib_glsl[];
extern char datatoc_common_math_lib_glsl[];
extern char datatoc_common_view_lib_glsl[];

extern char datatoc_eevee_camera_lib_glsl[];
extern char datatoc_eevee_film_filter_frag_glsl[];
extern char datatoc_eevee_film_lib_glsl[];
extern char datatoc_eevee_film_resolve_frag_glsl[];
extern char datatoc_eevee_object_forward_frag_glsl[];
extern char datatoc_eevee_object_lib_glsl[];
extern char datatoc_eevee_object_mesh_vert_glsl[];

extern char datatoc_eevee_shader_shared_hh[];

namespace blender::eevee {

/* Keep alphabetical order and clean prefix. */
enum eShaderType {
  FILM_FILTER = 0,
  FILM_RESOLVE,
  MESH, /* TEST */

  MAX_SHADER_TYPE,
};

typedef struct ShaderModule {
 private:
  struct ShaderDescription {
    const char *name;
    const char *vertex_shader_code;
    const char *geometry_shader_code;
    const char *fragment_shader_code;
  };

  DRWShaderLibrary *shader_lib_ = nullptr;
  std::array<GPUShader *, MAX_SHADER_TYPE> shaders_;
  std::array<ShaderDescription, MAX_SHADER_TYPE> shader_descriptions_;

 public:
  ShaderModule()
  {
    for (GPUShader *&shader : shaders_) {
      shader = nullptr;
    }

    shader_lib_ = DRW_shader_library_create();
    /* NOTE: These need to be ordered by dependencies. */
    DRW_shader_library_add_file(
        shader_lib_, datatoc_eevee_shader_shared_hh, "eevee_shader_shared.hh");
    DRW_SHADER_LIB_ADD(shader_lib_, common_math_lib);
    DRW_SHADER_LIB_ADD(shader_lib_, common_math_geom_lib);
    DRW_SHADER_LIB_ADD(shader_lib_, common_hair_lib);
    DRW_SHADER_LIB_ADD(shader_lib_, common_view_lib);
    DRW_SHADER_LIB_ADD(shader_lib_, eevee_camera_lib);
    DRW_SHADER_LIB_ADD(shader_lib_, eevee_film_lib);
    DRW_SHADER_LIB_ADD(shader_lib_, eevee_object_lib);

    /* Meh ¯\_(ツ)_/¯. */
    char *datatoc_nullptr_glsl = nullptr;

#define SHADER(enum_, vert_, geom_, frag_) \
  shader_descriptions_[enum_].name = STRINGIFY(enum_); \
  shader_descriptions_[enum_].vertex_shader_code = datatoc_##vert_##_glsl; \
  shader_descriptions_[enum_].geometry_shader_code = datatoc_##geom_##_glsl; \
  shader_descriptions_[enum_].fragment_shader_code = datatoc_##frag_##_glsl;

#define SHADER_FULLSCREEN(enum_, frag_) SHADER(enum_, common_fullscreen_vert, nullptr, frag_)

    SHADER_FULLSCREEN(FILM_FILTER, eevee_film_filter_frag);
    SHADER_FULLSCREEN(FILM_RESOLVE, eevee_film_resolve_frag);
    SHADER(MESH, eevee_object_mesh_vert, nullptr, eevee_object_forward_frag);

#undef SHADER
#undef SHADER_FULLSCREEN

#ifdef DEBUG
    /* Ensure all shader are described. */
    for (ShaderDescription &desc : shader_descriptions_) {
      BLI_assert(desc.name != nullptr);
      BLI_assert(desc.vertex_shader_code != nullptr);
      BLI_assert(desc.fragment_shader_code != nullptr);
    }
#endif
  }

  ~ShaderModule()
  {
    for (GPUShader *&shader : shaders_) {
      DRW_SHADER_FREE_SAFE(shader);
    }
    DRW_SHADER_LIB_FREE_SAFE(shader_lib_);
  }

  GPUShader *static_shader_get(eShaderType shader_type)
  {
    if (shaders_[shader_type] == nullptr) {
      ShaderDescription &desc = shader_descriptions_[shader_type];
      shaders_[shader_type] = DRW_shader_create_with_shaderlib_ex(desc.vertex_shader_code,
                                                                  desc.geometry_shader_code,
                                                                  desc.fragment_shader_code,
                                                                  shader_lib_,
                                                                  nullptr,
                                                                  desc.name);
      if (shaders_[shader_type] == nullptr) {
        fprintf(stderr, "EEVEE: error: Could not compile static shader \"%s\"\n", desc.name);
      }
      BLI_assert(shaders_[shader_type] != nullptr);
    }
    return shaders_[shader_type];
  }
} ShaderModule;

}  // namespace blender::eevee
