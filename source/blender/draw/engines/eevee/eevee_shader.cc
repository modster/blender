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

extern char datatoc_common_attribute_lib_glsl[];
extern char datatoc_common_fullscreen_vert_glsl[];
extern char datatoc_common_gpencil_lib_glsl[];
extern char datatoc_common_hair_lib_glsl[];
extern char datatoc_common_math_geom_lib_glsl[];
extern char datatoc_common_math_lib_glsl[];
extern char datatoc_common_obinfos_lib_glsl[];
extern char datatoc_common_uniform_attribute_lib_glsl[];
extern char datatoc_common_view_lib_glsl[];

extern char datatoc_eevee_bsdf_lib_glsl[];
extern char datatoc_eevee_bsdf_microfacet_lib_glsl[];
extern char datatoc_eevee_bsdf_sampling_lib_glsl[];
extern char datatoc_eevee_bsdf_stubs_lib_glsl[];
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
extern char datatoc_eevee_cubemap_lib_glsl[];
extern char datatoc_eevee_lightprobe_eval_cubemap_lib_glsl[];
extern char datatoc_eevee_lightprobe_eval_grid_lib_glsl[];
extern char datatoc_eevee_deferred_holdout_frag_glsl[];
extern char datatoc_eevee_deferred_transparent_frag_glsl[];
extern char datatoc_eevee_deferred_volume_frag_glsl[];
extern char datatoc_eevee_film_filter_frag_glsl[];
extern char datatoc_eevee_film_lib_glsl[];
extern char datatoc_eevee_film_resolve_frag_glsl[];
extern char datatoc_eevee_gbuffer_lib_glsl[];
extern char datatoc_eevee_irradiance_lib_glsl[];
extern char datatoc_eevee_light_lib_glsl[];
extern char datatoc_eevee_lightprobe_display_cubemap_frag_glsl[];
extern char datatoc_eevee_lightprobe_display_cubemap_vert_glsl[];
extern char datatoc_eevee_lightprobe_display_grid_frag_glsl[];
extern char datatoc_eevee_lightprobe_display_grid_vert_glsl[];
extern char datatoc_eevee_lightprobe_display_lib_glsl[];
extern char datatoc_eevee_lightprobe_filter_diffuse_frag_glsl[];
extern char datatoc_eevee_lightprobe_filter_downsample_frag_glsl[];
extern char datatoc_eevee_lightprobe_filter_geom_glsl[];
extern char datatoc_eevee_lightprobe_filter_glossy_frag_glsl[];
extern char datatoc_eevee_lightprobe_filter_lib_glsl[];
extern char datatoc_eevee_lightprobe_filter_vert_glsl[];
extern char datatoc_eevee_lightprobe_filter_visibility_frag_glsl[];
extern char datatoc_eevee_lookdev_background_frag_glsl[];
extern char datatoc_eevee_ltc_lib_glsl[];
extern char datatoc_eevee_motion_blur_gather_frag_glsl[];
extern char datatoc_eevee_motion_blur_lib_glsl[];
extern char datatoc_eevee_motion_blur_tiles_dilate_frag_glsl[];
extern char datatoc_eevee_motion_blur_tiles_flatten_frag_glsl[];
extern char datatoc_eevee_nodetree_eval_lib_glsl[];
extern char datatoc_eevee_sampling_lib_glsl[];
extern char datatoc_eevee_shadow_lib_glsl[];
extern char datatoc_eevee_surface_background_frag_glsl[];
extern char datatoc_eevee_surface_deferred_frag_glsl[];
extern char datatoc_eevee_surface_depth_simple_frag_glsl[];
extern char datatoc_eevee_surface_forward_frag_glsl[];
extern char datatoc_eevee_surface_lib_glsl[];
extern char datatoc_eevee_surface_mesh_vert_glsl[];
extern char datatoc_eevee_surface_gpencil_vert_glsl[];
extern char datatoc_eevee_surface_velocity_frag_glsl[];
extern char datatoc_eevee_surface_velocity_lib_glsl[];
extern char datatoc_eevee_surface_velocity_mesh_vert_glsl[];
extern char datatoc_eevee_surface_world_vert_glsl[];
extern char datatoc_eevee_velocity_lib_glsl[];
extern char datatoc_eevee_volume_deferred_frag_glsl[];
extern char datatoc_eevee_volume_eval_lib_glsl[];
extern char datatoc_eevee_volume_lib_glsl[];
extern char datatoc_eevee_volume_vert_glsl[];

extern char datatoc_eevee_shader_shared_hh[];

extern char datatoc_gpu_shader_codegen_lib_glsl[];

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
  DRW_SHADER_LIB_ADD(shader_lib_, common_attribute_lib);
  DRW_SHADER_LIB_ADD(shader_lib_, common_gpencil_lib);
  DRW_SHADER_LIB_ADD(shader_lib_, gpu_shader_codegen_lib);
  DRW_SHADER_LIB_ADD(shader_lib_, eevee_bsdf_lib);
  DRW_SHADER_LIB_ADD(shader_lib_, eevee_bsdf_microfacet_lib);
  DRW_SHADER_LIB_ADD(shader_lib_, eevee_bsdf_sampling_lib);
  DRW_SHADER_LIB_ADD(shader_lib_, eevee_bsdf_stubs_lib);
  DRW_SHADER_LIB_ADD(shader_lib_, eevee_irradiance_lib);
  DRW_SHADER_LIB_ADD(shader_lib_, eevee_closure_lib);
  DRW_SHADER_LIB_ADD(shader_lib_, eevee_cubemap_lib);
  DRW_SHADER_LIB_ADD(shader_lib_, eevee_gbuffer_lib);
  DRW_SHADER_LIB_ADD(shader_lib_, eevee_nodetree_eval_lib);
  DRW_SHADER_LIB_ADD(shader_lib_, eevee_sampling_lib);
  DRW_SHADER_LIB_ADD(shader_lib_, eevee_ltc_lib);
  DRW_SHADER_LIB_ADD(shader_lib_, eevee_shadow_lib);
  DRW_SHADER_LIB_ADD(shader_lib_, eevee_camera_lib);
  DRW_SHADER_LIB_ADD(shader_lib_, eevee_culling_lib);
  DRW_SHADER_LIB_ADD(shader_lib_, eevee_culling_iter_lib);
  DRW_SHADER_LIB_ADD(shader_lib_, eevee_light_lib);
  DRW_SHADER_LIB_ADD(shader_lib_, eevee_lightprobe_filter_lib);
  DRW_SHADER_LIB_ADD(shader_lib_, eevee_lightprobe_display_lib);
  DRW_SHADER_LIB_ADD(shader_lib_, eevee_lightprobe_eval_cubemap_lib);
  DRW_SHADER_LIB_ADD(shader_lib_, eevee_lightprobe_eval_grid_lib);
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
  SHADER(LIGHTPROBE_DISPLAY_CUBEMAP,
         eevee_lightprobe_display_cubemap_vert,
         nullptr,
         eevee_lightprobe_display_cubemap_frag,
         nullptr);
  SHADER(LIGHTPROBE_DISPLAY_IRRADIANCE,
         eevee_lightprobe_display_grid_vert,
         nullptr,
         eevee_lightprobe_display_grid_frag,
         nullptr);
  SHADER(LIGHTPROBE_FILTER_DOWNSAMPLE_CUBE,
         eevee_lightprobe_filter_vert,
         eevee_lightprobe_filter_geom,
         eevee_lightprobe_filter_downsample_frag,
         "#define CUBEMAP\n");
  SHADER(LIGHTPROBE_FILTER_GLOSSY,
         eevee_lightprobe_filter_vert,
         eevee_lightprobe_filter_geom,
         eevee_lightprobe_filter_glossy_frag,
         "#define CUBEMAP\n");
  SHADER(LIGHTPROBE_FILTER_DIFFUSE,
         eevee_lightprobe_filter_vert,
         eevee_lightprobe_filter_geom,
         eevee_lightprobe_filter_diffuse_frag,
         nullptr);
  SHADER(LIGHTPROBE_FILTER_VISIBILITY,
         eevee_lightprobe_filter_vert,
         eevee_lightprobe_filter_geom,
         eevee_lightprobe_filter_visibility_frag,
         nullptr);

  SHADER_FULLSCREEN(LOOKDEV_BACKGROUND, eevee_lookdev_background_frag);
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

char *ShaderModule::material_shader_code_defs_get(eMaterialGeometry geometry_type,
                                                  eMaterialDomain domain_type)
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
  switch (geometry_type) {
    case MAT_GEOM_GPENCIL:
      output += "#define UNIFORM_RESOURCE_ID\n";
      break;
    default:
      break;
  }

  return BLI_strdup(output.c_str());
}

char *ShaderModule::material_shader_code_vert_get(const GPUCodegenOutput *codegen,
                                                  GPUMaterial *mat,
                                                  eMaterialGeometry geometry_type)
{
  std::string output = "\n\n";

  /* Might be needed by attr_load_orco. */
  if (GPU_material_flag_get(mat, GPU_MATFLAG_OBJECT_INFO)) {
    output += datatoc_common_obinfos_lib_glsl;
  }

  if (codegen->attribs_interface) {
    /* Declare inputs. */
    std::string delimiter = ";\n";
    std::string sub(codegen->attribs_declare);
    size_t pos = 0;
    while ((pos = sub.find(delimiter)) != std::string::npos) {
      switch (geometry_type) {
        case MAT_GEOM_MESH:
          /* Example print:
           * in vec2 u015684; */
          output += "in ";
          output += sub.substr(0, pos + delimiter.length());
          break;
        case MAT_GEOM_HAIR:
          /* Example print:
           * uniform samplerBuffer u015684; */
          output += "uniform samplerBuffer ";
          output += sub.substr(sub.find(" ") + 1, pos + delimiter.length());
          break;
        case MAT_GEOM_GPENCIL:
          /* Example print:
           * vec2 u015684;
           * These are not used and just here to make the attribs_load functions call valids.
           * Only one uv and one color attribute layer is supported by gpencil objects. */
          output += sub.substr(0, pos + delimiter.length());
          break;
        case MAT_GEOM_WORLD:
        case MAT_GEOM_VOLUME:
          /* Not supported. */
          break;
      }
      sub.erase(0, pos + delimiter.length());
    }
    output += "\n";

    if (geometry_type != MAT_GEOM_WORLD) {
      output += "IN_OUT AttributesInterface\n";
      output += "{\n";
      output += codegen->attribs_interface;
      output += "};\n\n";
    }
  }

  output += "void attrib_load(void)\n";
  output += "{\n";
  if (codegen->attribs_load && geometry_type != MAT_GEOM_WORLD) {
    output += codegen->attribs_load;
  }
  output += "}\n\n";

  /* Displacement is only supported on mesh geometry for now. */
  if (geometry_type == MAT_GEOM_MESH) {
    if (codegen->displacement) {
      if (GPU_material_flag_get(mat, GPU_MATFLAG_UNIFORMS_ATTRIB)) {
        output += datatoc_common_uniform_attribute_lib_glsl;
      }
      output += codegen->uniforms;
      output += "\n";
      output += codegen->library;
      output += "\n";
    }

    output += "vec3 nodetree_displacement(void)\n";
    output += "{\n";
    if (codegen->displacement) {
      output += codegen->displacement;
    }
    else {
      output += "return vec3(0);\n";
    }
    output += "}\n\n";
  }

  switch (geometry_type) {
    case MAT_GEOM_WORLD:
      output += datatoc_eevee_surface_world_vert_glsl;
      break;
    case MAT_GEOM_VOLUME:
      output += datatoc_eevee_volume_vert_glsl;
      break;
    case MAT_GEOM_GPENCIL:
      output += datatoc_eevee_surface_gpencil_vert_glsl;
      break;
    case MAT_GEOM_MESH:
    default:
      output += datatoc_eevee_surface_mesh_vert_glsl;
      break;
  }

  return DRW_shader_library_create_shader_string(shader_lib_, output.c_str());
}

char *ShaderModule::material_shader_code_geom_get(const GPUCodegenOutput *UNUSED(codegen),
                                                  GPUMaterial *mat,
                                                  eMaterialGeometry geometry_type,
                                                  eMaterialDomain domain_type)
{
  /* Force geometry usage if GPU_BARYCENTRIC_DIST or GPU_BARYCENTRIC_TEXCO are used.
   * Note: GPU_BARYCENTRIC_TEXCO only requires it if the shader is not drawing hairs. */
  if ((geometry_type != MAT_GEOM_HAIR) && (domain_type == MAT_DOMAIN_SURFACE) &&
      GPU_material_flag_get(mat, GPU_MATFLAG_BARYCENTRIC)) {
    /* TODO. */
    BLI_assert(0);
  }
  return nullptr;
}

char *ShaderModule::material_shader_code_frag_get(const GPUCodegenOutput *codegen,
                                                  GPUMaterial *mat,
                                                  eMaterialGeometry geometry_type,
                                                  eMaterialPipeline pipeline_type)
{
  std::string output = "\n\n";

  /* World material loads attribs in fragment shader (only used for orco). */
  if (geometry_type == MAT_GEOM_WORLD) {
    if (codegen->attribs_interface) {
      /* Declare inputs. */
      std::string delimiter = ";\n";
      std::string sub(codegen->attribs_declare);
      size_t pos = 0;
      while ((pos = sub.find(delimiter)) != std::string::npos) {
        /* Example print:
         * vec2 u015684;
         * These are not used and just here to make the attribs_load functions call valids.
         * Only orco layer is supported by world. */
        output += sub.substr(0, pos + delimiter.length());
        sub.erase(0, pos + delimiter.length());
      }
      output += "\n";

      output += codegen->attribs_interface;
      output += "\n";
    }

    output += "void attrib_load(void)\n";
    output += "{\n";
    if (codegen->attribs_interface) {
      output += codegen->attribs_load;
    }
    output += "}\n\n";
  }
  else {
    if (codegen->attribs_interface) {
      output += "IN_OUT AttributesInterface\n";
      output += "{\n";
      output += codegen->attribs_interface;
      output += "};\n\n";
    }
  }

  if (codegen->surface || codegen->volume) {
    if (GPU_material_flag_get(mat, GPU_MATFLAG_UNIFORMS_ATTRIB)) {
      output += datatoc_common_uniform_attribute_lib_glsl;
    }
    if (GPU_material_flag_get(mat, GPU_MATFLAG_OBJECT_INFO)) {
      output += datatoc_common_obinfos_lib_glsl;
    }
    output += codegen->uniforms;
    output += "\n";
    output += codegen->library;
    output += "\n";
  }

  output += "Closure nodetree_surface(void)\n";
  output += "{\n";
  if (codegen->surface) {
    output += codegen->surface;
  }
  else {
    output += "return CLOSURE_DEFAULT;\n";
  }
  output += "}\n\n";

  output += "Closure nodetree_volume(void)\n";
  output += "{\n";
  if (codegen->volume) {
    output += codegen->volume;
  }
  else {
    output += "return CLOSURE_DEFAULT;\n";
  }
  output += "}\n\n";

  switch (geometry_type) {
    case MAT_GEOM_WORLD:
      output += datatoc_eevee_surface_background_frag_glsl;
      break;
    case MAT_GEOM_VOLUME:
      if (pipeline_type == MAT_PIPE_DEFERRED) {
        output += datatoc_eevee_volume_deferred_frag_glsl;
      }
      else {
        output += nullptr; /* TODO */
      }
      break;
    default:
      if (pipeline_type == MAT_PIPE_DEFERRED) {
        output += datatoc_eevee_surface_deferred_frag_glsl;
      }
      else {
        output += datatoc_eevee_surface_forward_frag_glsl;
      }
      break;
  }

  return DRW_shader_library_create_shader_string(shader_lib_, output.c_str());
}

/* WATCH: This can be called from another thread! Needs to not touch the shader module in any
 * thread unsafe manner. */
GPUShaderSource ShaderModule::material_shader_code_generate(GPUMaterial *mat,
                                                            const GPUCodegenOutput *codegen)
{
  uint64_t shader_uuid = GPU_material_uuid_get(mat);

  eMaterialPipeline pipeline_type;
  eMaterialGeometry geometry_type;
  eMaterialDomain domain_type;
  material_type_from_shader_uuid(shader_uuid, pipeline_type, geometry_type, domain_type);

  GPUShaderSource source;
  source.vertex = material_shader_code_vert_get(codegen, mat, geometry_type);
  source.fragment = material_shader_code_frag_get(codegen, mat, geometry_type, pipeline_type);
  source.geometry = material_shader_code_geom_get(codegen, mat, geometry_type, domain_type);
  source.defines = material_shader_code_defs_get(geometry_type, domain_type);
  return source;
}

static GPUShaderSource codegen_callback(void *thunk,
                                        GPUMaterial *mat,
                                        const GPUCodegenOutput *codegen)
{
  return ((ShaderModule *)thunk)->material_shader_code_generate(mat, codegen);
}

GPUMaterial *ShaderModule::material_shader_get(::Material *blender_mat,
                                               struct bNodeTree *nodetree,
                                               eMaterialGeometry geometry_type,
                                               eMaterialDomain domain_type,
                                               bool deferred_compilation)
{
  /* TODO derive from mat. */
  eMaterialPipeline pipeline_type = MAT_PIPE_DEFERRED;

  uint64_t shader_uuid = shader_uuid_from_material_type(pipeline_type, geometry_type, domain_type);

  bool is_volume = (domain_type == MAT_DOMAIN_VOLUME);

  return DRW_shader_from_material(
      blender_mat, nodetree, shader_uuid, is_volume, deferred_compilation, codegen_callback, this);
}

GPUMaterial *ShaderModule::world_shader_get(::World *blender_world,
                                            struct bNodeTree *nodetree,
                                            eMaterialDomain domain_type)
{
  eMaterialPipeline pipeline_type = MAT_PIPE_DEFERRED; /* Unused. */
  eMaterialGeometry geometry_type = MAT_GEOM_WORLD;

  uint64_t shader_uuid = shader_uuid_from_material_type(pipeline_type, geometry_type, domain_type);

  bool is_volume = (domain_type == MAT_DOMAIN_VOLUME);
  bool deferred_compilation = false;

  return DRW_shader_from_world(blender_world,
                               nodetree,
                               shader_uuid,
                               is_volume,
                               deferred_compilation,
                               codegen_callback,
                               this);
}

/* Variation to compile a material only with a nodetree. Caller needs to maintain the list of
 * materials and call GPU_material_free on it to update the material. */
GPUMaterial *ShaderModule::material_shader_get(const char *name,
                                               ListBase &materials,
                                               struct bNodeTree *nodetree,
                                               eMaterialGeometry geometry_type,
                                               eMaterialDomain domain_type,
                                               bool is_lookdev)
{
  eMaterialPipeline pipeline_type = MAT_PIPE_DEFERRED; /* Unused. */

  uint64_t shader_uuid = shader_uuid_from_material_type(pipeline_type, geometry_type, domain_type);

  bool is_volume = (domain_type == MAT_DOMAIN_VOLUME);

  GPUMaterial *gpumat = GPU_material_from_nodetree(nullptr,
                                                   nullptr,
                                                   nodetree,
                                                   &materials,
                                                   name,
                                                   shader_uuid,
                                                   is_volume,
                                                   is_lookdev,
                                                   codegen_callback,
                                                   this);
  GPU_material_status_set(gpumat, GPU_MAT_QUEUED);
  GPU_material_compile(gpumat);
  return gpumat;
}

/** \} */

}  // namespace blender::eevee
