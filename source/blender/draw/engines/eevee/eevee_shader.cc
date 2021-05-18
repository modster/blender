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

#include "eevee_shader.hh"
#include "eevee_material.hh"

extern char datatoc_common_fullscreen_vert_glsl[];
extern char datatoc_common_hair_lib_glsl[];
extern char datatoc_common_math_geom_lib_glsl[];
extern char datatoc_common_math_lib_glsl[];
extern char datatoc_common_view_lib_glsl[];

extern char datatoc_eevee_camera_lib_glsl[];
extern char datatoc_eevee_camera_velocity_frag_glsl[];
extern char datatoc_eevee_closure_lib_glsl[];
extern char datatoc_eevee_culling_debug_frag_glsl[];
extern char datatoc_eevee_culling_iter_lib_glsl[];
extern char datatoc_eevee_culling_lib_glsl[];
extern char datatoc_eevee_culling_light_frag_glsl[];
extern char datatoc_eevee_depth_clear_frag_glsl[];
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
extern char datatoc_eevee_deferred_direct_frag_glsl[];
extern char datatoc_eevee_deferred_holdout_frag_glsl[];
extern char datatoc_eevee_deferred_transparent_frag_glsl[];
extern char datatoc_eevee_deferred_volume_frag_glsl[];
extern char datatoc_eevee_film_filter_frag_glsl[];
extern char datatoc_eevee_film_lib_glsl[];
extern char datatoc_eevee_film_resolve_frag_glsl[];
extern char datatoc_eevee_gbuffer_lib_glsl[];
extern char datatoc_eevee_light_lib_glsl[];
extern char datatoc_eevee_ltc_lib_glsl[];
extern char datatoc_eevee_motion_blur_gather_frag_glsl[];
extern char datatoc_eevee_motion_blur_lib_glsl[];
extern char datatoc_eevee_motion_blur_tiles_dilate_frag_glsl[];
extern char datatoc_eevee_motion_blur_tiles_flatten_frag_glsl[];
extern char datatoc_eevee_nodetree_eval_lib_glsl[];
extern char datatoc_eevee_sampling_lib_glsl[];
extern char datatoc_eevee_shadow_lib_glsl[];
extern char datatoc_eevee_surface_deferred_frag_glsl[];
extern char datatoc_eevee_surface_depth_simple_frag_glsl[];
extern char datatoc_eevee_surface_forward_frag_glsl[];
extern char datatoc_eevee_surface_lib_glsl[];
extern char datatoc_eevee_surface_mesh_vert_glsl[];
extern char datatoc_eevee_surface_velocity_frag_glsl[];
extern char datatoc_eevee_surface_velocity_lib_glsl[];
extern char datatoc_eevee_surface_velocity_mesh_vert_glsl[];
extern char datatoc_eevee_velocity_lib_glsl[];
extern char datatoc_eevee_volume_deferred_frag_glsl[];
extern char datatoc_eevee_volume_eval_lib_glsl[];
extern char datatoc_eevee_volume_lib_glsl[];
extern char datatoc_eevee_volume_vert_glsl[];

extern char datatoc_eevee_shader_shared_hh[];

namespace blender::eevee {

/** \} */

/* -------------------------------------------------------------------- */
/** \name Static shaders
 *
 * \{ */

ShaderModule::ShaderModule()
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
  DRW_SHADER_LIB_ADD(shader_lib_, eevee_closure_lib);
  DRW_SHADER_LIB_ADD(shader_lib_, eevee_gbuffer_lib);
  DRW_SHADER_LIB_ADD(shader_lib_, eevee_nodetree_eval_lib);
  DRW_SHADER_LIB_ADD(shader_lib_, eevee_sampling_lib);
  DRW_SHADER_LIB_ADD(shader_lib_, eevee_ltc_lib);
  DRW_SHADER_LIB_ADD(shader_lib_, eevee_shadow_lib);
  DRW_SHADER_LIB_ADD(shader_lib_, eevee_camera_lib);
  DRW_SHADER_LIB_ADD(shader_lib_, eevee_culling_lib);
  DRW_SHADER_LIB_ADD(shader_lib_, eevee_culling_iter_lib);
  DRW_SHADER_LIB_ADD(shader_lib_, eevee_light_lib);
  DRW_SHADER_LIB_ADD(shader_lib_, eevee_volume_eval_lib);
  DRW_SHADER_LIB_ADD(shader_lib_, eevee_volume_lib);
  DRW_SHADER_LIB_ADD(shader_lib_, eevee_velocity_lib);
  DRW_SHADER_LIB_ADD(shader_lib_, eevee_depth_of_field_lib);
  DRW_SHADER_LIB_ADD(shader_lib_, eevee_depth_of_field_accumulator_lib);
  DRW_SHADER_LIB_ADD(shader_lib_, eevee_depth_of_field_scatter_lib);
  DRW_SHADER_LIB_ADD(shader_lib_, eevee_film_lib);
  DRW_SHADER_LIB_ADD(shader_lib_, eevee_motion_blur_lib);
  DRW_SHADER_LIB_ADD(shader_lib_, eevee_surface_lib);
  DRW_SHADER_LIB_ADD(shader_lib_, eevee_surface_velocity_lib);

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

  SHADER_FULLSCREEN(CULLING_DEBUG, eevee_culling_debug_frag);
  SHADER_FULLSCREEN(CULLING_LIGHT, eevee_culling_light_frag);
  SHADER_FULLSCREEN(FILM_FILTER, eevee_film_filter_frag);
  SHADER_FULLSCREEN(FILM_RESOLVE, eevee_film_resolve_frag);
  SHADER_FULLSCREEN(DEFERRED_EVAL_DIRECT, eevee_deferred_direct_frag);
  SHADER_FULLSCREEN(DEFERRED_EVAL_HOLDOUT, eevee_deferred_holdout_frag);
  SHADER_FULLSCREEN(DEFERRED_EVAL_TRANSPARENT, eevee_deferred_transparent_frag);
  SHADER_FULLSCREEN(DEFERRED_EVAL_VOLUME, eevee_deferred_volume_frag);
  SHADER(DEFERRED_MESH, eevee_surface_mesh_vert, nullptr, eevee_surface_deferred_frag, nullptr);
  SHADER(DEFERRED_VOLUME, eevee_volume_vert, nullptr, eevee_volume_deferred_frag, nullptr);
  SHADER(DEPTH_SIMPLE_MESH,
         eevee_surface_mesh_vert,
         nullptr,
         eevee_surface_depth_simple_frag,
         nullptr);
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
  SHADER(MESH, eevee_surface_mesh_vert, nullptr, eevee_surface_forward_frag, nullptr);

  SHADER_FULLSCREEN(MOTION_BLUR_GATHER, eevee_motion_blur_gather_frag);
  SHADER_FULLSCREEN(MOTION_BLUR_TILE_DILATE, eevee_motion_blur_tiles_dilate_frag);
  SHADER_FULLSCREEN(MOTION_BLUR_TILE_FLATTEN, eevee_motion_blur_tiles_flatten_frag);

  SHADER_FULLSCREEN(SHADOW_CLEAR, eevee_depth_clear_frag);

  SHADER(VELOCITY_MESH,
         eevee_surface_velocity_mesh_vert,
         nullptr,
         eevee_surface_velocity_frag,
         nullptr);
  SHADER_FULLSCREEN(VELOCITY_CAMERA, eevee_camera_velocity_frag);

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

ShaderModule::~ShaderModule()
{
  for (GPUShader *&shader : shaders_) {
    DRW_SHADER_FREE_SAFE(shader);
  }
  DRW_SHADER_LIB_FREE_SAFE(shader_lib_);
}

GPUShader *ShaderModule::static_shader_get(eShaderType shader_type)
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

/* Run some custom preprocessor shader rewrite and returns a new string. */
std::string ShaderModule::enum_preprocess(const char *input)
{
  std::string output = "";
  /* Not failure safe but this only runs on static data. */
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

/** \} */

/* -------------------------------------------------------------------- */
/** \name GPU Materials
 *
 * \{ */

std::string ShaderModule::material_shader_code_defs_get(eMaterialDomain domain_type)
{
  std::string output = "";

  switch (domain_type) {
    case MAT_DOMAIN_VOLUME:
      output += "#define VOLUME_SHADER\n";
      break;
    default:
    case MAT_DOMAIN_SURFACE:
      /* TODO rename define or make it obsolete. */
      output += "#define MESH_SHADER\n";
      break;
  }
  return output;
}

char *ShaderModule::material_shader_code_vert_get(eMaterialGeometry geometry_type)
{
  switch (geometry_type) {
    case MAT_GEOM_VOLUME:
      return DRW_shader_library_create_shader_string(shader_lib_, datatoc_eevee_volume_vert_glsl);
    default:
      return DRW_shader_library_create_shader_string(shader_lib_,
                                                     datatoc_eevee_surface_mesh_vert_glsl);
  }
}

char *ShaderModule::material_shader_code_frag_get(eMaterialGeometry geometry_type,
                                                  eMaterialPipeline pipeline_type)
{
  char const *frag_code;

  switch (geometry_type) {
    case MAT_GEOM_VOLUME:
      frag_code = (pipeline_type == MAT_PIPE_DEFERRED) ? datatoc_eevee_volume_deferred_frag_glsl :
                                                         nullptr; /* TODO */
      break;
    default:
      frag_code = (pipeline_type == MAT_PIPE_DEFERRED) ? datatoc_eevee_surface_deferred_frag_glsl :
                                                         datatoc_eevee_surface_forward_frag_glsl;
      break;
  }

  return DRW_shader_library_create_shader_string(shader_lib_, frag_code);
}

static void material_post_eval(GPUMaterial *mat,
                               int options,
                               const char **UNUSED(vert_code),
                               const char **geom_code,
                               const char **UNUSED(frag_lib),
                               const char **UNUSED(defines))
{
  const bool is_hair = (options & MAT_GEOM_HAIR) != 0;
  const bool is_surface = (options & MAT_DOMAIN_SURFACE) != 0;

  /* Force geometry usage if GPU_BARYCENTRIC_DIST or GPU_BARYCENTRIC_TEXCO are used.
   * Note: GPU_BARYCENTRIC_TEXCO only requires it if the shader is not drawing hairs. */
  if (!is_hair && is_surface && GPU_material_flag_get(mat, GPU_MATFLAG_BARYCENTRIC) &&
      *geom_code == NULL) {
    /* TODO. */
    BLI_assert(0);
  }
}

GPUMaterial *ShaderModule::material_shader_get(Scene *scene,
                                               ::Material *blender_mat,
                                               eMaterialGeometry geometry_type,
                                               eMaterialDomain domain_type,
                                               bool deferred_compilation)
{
  /* This work for now as we share the material module with all instance. */
  void *engine_id = (void *)&DRW_engine_viewport_eevee_type;

  /* TODO derive from mat. */
  eMaterialPipeline pipeline_type = MAT_PIPE_DEFERRED;

  int options = geometry_type | (domain_type << 2) | (pipeline_type << 3);

  GPUMaterial *gpumat = GPU_material_from_nodetree_find(
      &blender_mat->gpumaterial, engine_id, options);
  if (gpumat) {
    return gpumat;
  }

  bool is_volume = (domain_type == MAT_DOMAIN_SURFACE);

  char *vert_code = material_shader_code_vert_get(geometry_type);
  char *frag_code = material_shader_code_frag_get(geometry_type, pipeline_type);
  std::string defines_code = material_shader_code_defs_get(domain_type);

  gpumat = DRW_shader_create_from_material(scene,
                                           blender_mat,
                                           blender_mat->nodetree,
                                           engine_id,
                                           options,
                                           is_volume,
                                           vert_code,
                                           nullptr,
                                           frag_code,
                                           defines_code.c_str(),
                                           deferred_compilation,
                                           material_post_eval);

  MEM_SAFE_FREE(vert_code);
  MEM_SAFE_FREE(frag_code);

  return gpumat;
}

/** \} */

}  // namespace blender::eevee
