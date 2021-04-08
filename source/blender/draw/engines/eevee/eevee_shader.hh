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
#include <string>

#include "BLI_string_ref.hh"
#include "DRW_render.h"
#include "GPU_shader.h"

extern char datatoc_common_fullscreen_vert_glsl[];
extern char datatoc_common_hair_lib_glsl[];
extern char datatoc_common_math_geom_lib_glsl[];
extern char datatoc_common_math_lib_glsl[];
extern char datatoc_common_view_lib_glsl[];

extern char datatoc_eevee_camera_lib_glsl[];
extern char datatoc_eevee_depth_of_field_accumulator_lib_glsl[];
extern char datatoc_eevee_depth_of_field_bokeh_lut_frag_glsl[];
extern char datatoc_eevee_depth_of_field_downsample_frag_glsl[];
extern char datatoc_eevee_depth_of_field_filter_frag_glsl[];
extern char datatoc_eevee_depth_of_field_gather_frag_glsl[];
extern char datatoc_eevee_depth_of_field_gather_holefill_frag_glsl[];
extern char datatoc_eevee_depth_of_field_lib_glsl[];
extern char datatoc_eevee_depth_of_field_reduce_copy_frag_glsl[];
extern char datatoc_eevee_depth_of_field_reduce_downsample_frag_glsl[];
extern char datatoc_eevee_depth_of_field_reduce_recursive_frag_glsl[];
extern char datatoc_eevee_depth_of_field_resolve_frag_glsl[];
extern char datatoc_eevee_depth_of_field_scatter_frag_glsl[];
extern char datatoc_eevee_depth_of_field_scatter_lib_glsl[];
extern char datatoc_eevee_depth_of_field_scatter_vert_glsl[];
extern char datatoc_eevee_depth_of_field_setup_frag_glsl[];
extern char datatoc_eevee_depth_of_field_tiles_dilate_frag_glsl[];
extern char datatoc_eevee_depth_of_field_tiles_flatten_frag_glsl[];
extern char datatoc_eevee_film_filter_frag_glsl[];
extern char datatoc_eevee_film_lib_glsl[];
extern char datatoc_eevee_film_resolve_frag_glsl[];
extern char datatoc_eevee_object_forward_frag_glsl[];
extern char datatoc_eevee_object_lib_glsl[];
extern char datatoc_eevee_object_mesh_vert_glsl[];
extern char datatoc_eevee_sampling_lib_glsl[];

extern char datatoc_eevee_shader_shared_hh[];

namespace blender::eevee {

/* Keep alphabetical order and clean prefix. */
enum eShaderType {
  DOF_BOKEH_LUT = 0,
  DOF_GATHER_BACKGROUND_LUT,
  DOF_GATHER_BACKGROUND,
  DOF_FILTER,
  DOF_GATHER_FOREGROUND_LUT,
  DOF_GATHER_FOREGROUND,
  DOF_GATHER_HOLEFILL,
  DOF_REDUCE_COPY,
  DOF_REDUCE_DOWNSAMPLE,
  DOF_REDUCE_RECURSIVE,
  DOF_RESOLVE,
  DOF_RESOLVE_HQ,
  DOF_RESOLVE_LUT,
  DOF_RESOLVE_LUT_HQ,
  DOF_SCATTER_BACKGROUND_LUT,
  DOF_SCATTER_BACKGROUND,
  DOF_SCATTER_FOREGROUND_LUT,
  DOF_SCATTER_FOREGROUND,
  DOF_SETUP,
  DOF_TILES_DILATE_MINABS,
  DOF_TILES_DILATE_MINMAX,
  DOF_TILES_FLATTEN,

  FILM_FILTER,
  FILM_RESOLVE,

  MESH, /* TEST */

  MAX_SHADER_TYPE,
};

class ShaderModule {
 private:
  struct ShaderDescription {
    const char *name;
    const char *vertex_shader_code;
    const char *geometry_shader_code;
    const char *fragment_shader_code;
    const char *defines_shader_code;
  };

  DRWShaderLibrary *shader_lib_ = nullptr;
  std::array<GPUShader *, MAX_SHADER_TYPE> shaders_;
  std::array<ShaderDescription, MAX_SHADER_TYPE> shader_descriptions_;
  std::string shared_lib_;

 public:
  ShaderModule()
  {
    for (GPUShader *&shader : shaders_) {
      shader = nullptr;
    }

    shared_lib_ = enum_preprocess(datatoc_eevee_shader_shared_hh);

    shader_lib_ = DRW_shader_library_create();
    /* NOTE: These need to be ordered by dependencies. */
    DRW_SHADER_LIB_ADD(shader_lib_, common_math_lib);
    DRW_shader_library_add_file(shader_lib_, shared_lib_.c_str(), "eevee_shader_shared.hh");
    DRW_SHADER_LIB_ADD(shader_lib_, common_math_geom_lib);
    DRW_SHADER_LIB_ADD(shader_lib_, common_hair_lib);
    DRW_SHADER_LIB_ADD(shader_lib_, common_view_lib);
    DRW_SHADER_LIB_ADD(shader_lib_, eevee_sampling_lib);
    DRW_SHADER_LIB_ADD(shader_lib_, eevee_camera_lib);
    DRW_SHADER_LIB_ADD(shader_lib_, eevee_depth_of_field_lib);
    DRW_SHADER_LIB_ADD(shader_lib_, eevee_depth_of_field_accumulator_lib);
    DRW_SHADER_LIB_ADD(shader_lib_, eevee_depth_of_field_scatter_lib);
    DRW_SHADER_LIB_ADD(shader_lib_, eevee_film_lib);
    DRW_SHADER_LIB_ADD(shader_lib_, eevee_object_lib);

    /* Meh ¯\_(ツ)_/¯. */
    char *datatoc_nullptr_glsl = nullptr;

#define SHADER(enum_, vert_, geom_, frag_, defs_) \
  shader_descriptions_[enum_].name = STRINGIFY(enum_); \
  shader_descriptions_[enum_].vertex_shader_code = datatoc_##vert_##_glsl; \
  shader_descriptions_[enum_].geometry_shader_code = datatoc_##geom_##_glsl; \
  shader_descriptions_[enum_].fragment_shader_code = datatoc_##frag_##_glsl; \
  shader_descriptions_[enum_].defines_shader_code = defs_;

#define SHADER_FULLSCREEN_DEFINES(enum_, frag_, defs_) \
  SHADER(enum_, common_fullscreen_vert, nullptr, frag_, defs_)
#define SHADER_FULLSCREEN(enum_, frag_) SHADER_FULLSCREEN_DEFINES(enum_, frag_, nullptr)

    SHADER_FULLSCREEN(FILM_FILTER, eevee_film_filter_frag);
    SHADER_FULLSCREEN(FILM_RESOLVE, eevee_film_resolve_frag);
    SHADER_FULLSCREEN(DOF_BOKEH_LUT, eevee_depth_of_field_bokeh_lut_frag);
    SHADER_FULLSCREEN(DOF_FILTER, eevee_depth_of_field_filter_frag);
    SHADER_FULLSCREEN_DEFINES(DOF_GATHER_BACKGROUND_LUT,
                              eevee_depth_of_field_gather_frag,
                              "#define DOF_FOREGROUND_PASS false\n"
                              "#define DOF_BOKEH_TEXTURE true\n");
    SHADER_FULLSCREEN_DEFINES(DOF_GATHER_BACKGROUND,
                              eevee_depth_of_field_gather_frag,
                              "#define DOF_FOREGROUND_PASS false\n"
                              "#define DOF_BOKEH_TEXTURE false\n");
    SHADER_FULLSCREEN_DEFINES(DOF_GATHER_FOREGROUND_LUT,
                              eevee_depth_of_field_gather_frag,
                              "#define DOF_FOREGROUND_PASS true\n"
                              "#define DOF_BOKEH_TEXTURE true\n");
    SHADER_FULLSCREEN_DEFINES(DOF_GATHER_FOREGROUND,
                              eevee_depth_of_field_gather_frag,
                              "#define DOF_FOREGROUND_PASS true\n"
                              "#define DOF_BOKEH_TEXTURE false\n");
    SHADER_FULLSCREEN_DEFINES(DOF_GATHER_HOLEFILL,
                              eevee_depth_of_field_gather_holefill_frag,
                              "#define DOF_HOLEFILL_PASS true\n"
                              "#define DOF_FOREGROUND_PASS false\n"
                              "#define DOF_BOKEH_TEXTURE false\n");
    SHADER_FULLSCREEN(DOF_REDUCE_COPY, eevee_depth_of_field_reduce_copy_frag);
    SHADER_FULLSCREEN(DOF_REDUCE_DOWNSAMPLE, eevee_depth_of_field_reduce_downsample_frag);
    SHADER_FULLSCREEN(DOF_REDUCE_RECURSIVE, eevee_depth_of_field_reduce_recursive_frag);
    SHADER_FULLSCREEN_DEFINES(DOF_RESOLVE_LUT,
                              eevee_depth_of_field_resolve_frag,
                              "#define DOF_RESOLVE_PASS true\n"
                              "#define DOF_BOKEH_TEXTURE true\n"
                              "#define DOF_SLIGHT_FOCUS_DENSITY 2\n");
    SHADER_FULLSCREEN_DEFINES(DOF_RESOLVE_LUT_HQ,
                              eevee_depth_of_field_resolve_frag,
                              "#define DOF_RESOLVE_PASS true\n"
                              "#define DOF_BOKEH_TEXTURE true\n"
                              "#define DOF_SLIGHT_FOCUS_DENSITY 4\n");
    SHADER_FULLSCREEN_DEFINES(DOF_RESOLVE,
                              eevee_depth_of_field_resolve_frag,
                              "#define DOF_RESOLVE_PASS true\n"
                              "#define DOF_BOKEH_TEXTURE false\n"
                              "#define DOF_SLIGHT_FOCUS_DENSITY 2\n");
    SHADER_FULLSCREEN_DEFINES(DOF_RESOLVE_HQ,
                              eevee_depth_of_field_resolve_frag,
                              "#define DOF_RESOLVE_PASS true\n"
                              "#define DOF_BOKEH_TEXTURE false\n"
                              "#define DOF_SLIGHT_FOCUS_DENSITY 4\n");
    SHADER(DOF_SCATTER_BACKGROUND_LUT,
           eevee_depth_of_field_scatter_vert,
           nullptr,
           eevee_depth_of_field_scatter_frag,
           "#define DOF_FOREGROUND_PASS false\n"
           "#define DOF_BOKEH_TEXTURE true\n");
    SHADER(DOF_SCATTER_BACKGROUND,
           eevee_depth_of_field_scatter_vert,
           nullptr,
           eevee_depth_of_field_scatter_frag,
           "#define DOF_FOREGROUND_PASS false\n"
           "#define DOF_BOKEH_TEXTURE false\n");
    SHADER(DOF_SCATTER_FOREGROUND_LUT,
           eevee_depth_of_field_scatter_vert,
           nullptr,
           eevee_depth_of_field_scatter_frag,
           "#define DOF_FOREGROUND_PASS true\n"
           "#define DOF_BOKEH_TEXTURE true\n");
    SHADER(DOF_SCATTER_FOREGROUND,
           eevee_depth_of_field_scatter_vert,
           nullptr,
           eevee_depth_of_field_scatter_frag,
           "#define DOF_FOREGROUND_PASS true\n"
           "#define DOF_BOKEH_TEXTURE false\n");
    SHADER_FULLSCREEN(DOF_SETUP, eevee_depth_of_field_setup_frag);
    SHADER_FULLSCREEN_DEFINES(DOF_TILES_DILATE_MINABS,
                              eevee_depth_of_field_tiles_dilate_frag,
                              "#define DILATE_MODE_MIN_MAX false\n");
    SHADER_FULLSCREEN_DEFINES(DOF_TILES_DILATE_MINMAX,
                              eevee_depth_of_field_tiles_dilate_frag,
                              "#define DILATE_MODE_MIN_MAX true\n");
    SHADER_FULLSCREEN(DOF_TILES_FLATTEN, eevee_depth_of_field_tiles_flatten_frag);
    SHADER(MESH, eevee_object_mesh_vert, nullptr, eevee_object_forward_frag, nullptr);

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
                                                                  desc.defines_shader_code,
                                                                  desc.name);
      if (shaders_[shader_type] == nullptr) {
        fprintf(stderr, "EEVEE: error: Could not compile static shader \"%s\"\n", desc.name);
      }
      BLI_assert(shaders_[shader_type] != nullptr);
    }
    return shaders_[shader_type];
  }

 private:
  /* Run some custom preprocessor shader rewrite and returns a new string. */
  std::string enum_preprocess(const char *input)
  {
    std::string output = "";
    /* Not failure safe but this is only ran on static data. */
    const char *cursor = input;
    while ((cursor = strstr(cursor, "enum "))) {
      output += StringRef(input, cursor - input);

      /* Skip "enum" keyword. */
      cursor = strstr(cursor, " ");

      const char *enum_name = cursor;
      cursor = strstr(cursor, " :");

      output += "#define " + StringRef(enum_name, cursor - enum_name) + " uint\n";
      output += "const uint ";

      const char *enum_values = strstr(cursor, "{") + 1;
      cursor = strstr(cursor, "}");
      output += StringRef(enum_values, cursor - enum_values);

      if (cursor != nullptr) {
        /* Skip the curly bracket but not the semicolon. */
        input = cursor + 1;
      }
      else {
        input = nullptr;
      }
    }
    if (input != nullptr) {
      output += input;
    }
    return output;
  }
};

}  // namespace blender::eevee
