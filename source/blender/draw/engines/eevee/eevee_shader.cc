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

#ifdef DEBUG
  /* Ensure all shader are described. */
  for (auto i : IndexRange(MAX_SHADER_TYPE)) {
    const char *name = static_shader_create_info_name_get(eShaderType(i));
    if (name == nullptr) {
      std::cerr << "EEVEE: Missing case for eShaderType(" << i
                << ") in static_shader_create_info_name_get().";
      BLI_assert(0);
    }
    const GPUShaderCreateInfo *create_info = GPU_shader_create_info_get(name);
    BLI_assert_msg(create_info != nullptr, "EEVEE: Missing create info for static shader.");
  }
#endif
}

ShaderModule::~ShaderModule()
{
  for (GPUShader *&shader : shaders_) {
    DRW_SHADER_FREE_SAFE(shader);
  }
}

const char *ShaderModule::static_shader_create_info_name_get(eShaderType shader_type)
{
  switch (shader_type) {
    case CULLING_DEBUG:
      return "eevee_culling_debug";
    case CULLING_SELECT:
      return "eevee_culling_select";
    case CULLING_SORT:
      return "eevee_culling_sort";
    case CULLING_TILE:
      return "eevee_culling_tile";

    case FILM_FILTER:
      return "eevee_film_filter";
    case FILM_RESOLVE:
      return "eevee_film_resolve";
    case FILM_RESOLVE_DEPTH:
      return "eevee_film_resolve_depth";

    case DEFERRED_EVAL_DIRECT:
      return "eevee_deferred_direct";
    case DEFERRED_EVAL_HOLDOUT:
      return "eevee_deferred_holdout";
    case DEFERRED_EVAL_TRANSPARENT:
      return "eevee_deferred_transparent";
    case DEFERRED_EVAL_VOLUME:
      return "eevee_deferred_volume";

    case DEFERRED_VOLUME:
      return "eevee_volume_deferred";

    case HIZ_COPY:
      return "eevee_hiz_copy";
    case HIZ_DOWNSAMPLE:
      return "eevee_hiz_downsample";

    case DOF_BOKEH_LUT:
      return "eevee_depth_of_field_bokeh_lut";
    case DOF_FILTER:
      return "eevee_depth_of_field_filter";
    case DOF_GATHER_BACKGROUND_LUT:
      return "eevee_depth_of_field_gather_background_lut";
    case DOF_GATHER_BACKGROUND:
      return "eevee_depth_of_field_gather_background";
    case DOF_GATHER_FOREGROUND_LUT:
      return "eevee_depth_of_field_gather_foreground_lut";
    case DOF_GATHER_FOREGROUND:
      return "eevee_depth_of_field_gather_foreground";
    case DOF_GATHER_HOLEFILL:
      return "eevee_depth_of_field_gather_holefill";
    case DOF_REDUCE_COPY:
      return "eevee_depth_of_field_reduce_copy";
    case DOF_REDUCE_DOWNSAMPLE:
      return "eevee_depth_of_field_reduce_downsample";
    case DOF_REDUCE_RECURSIVE:
      return "eevee_depth_of_field_reduce_recursive";
    case DOF_RESOLVE_LUT:
      return "eevee_depth_of_field_resolve_lut";
    case DOF_RESOLVE_LUT_HQ:
      return "eevee_depth_of_field_resolve_lut_hq";
    case DOF_RESOLVE:
      return "eevee_depth_of_field_resolve";
    case DOF_RESOLVE_HQ:
      return "eevee_depth_of_field_resolve_hq";
    case DOF_SCATTER_BACKGROUND_LUT:
      return "eevee_depth_of_field_scatter_background_lut";
    case DOF_SCATTER_BACKGROUND:
      return "eevee_depth_of_field_scatter_background";
    case DOF_SCATTER_FOREGROUND_LUT:
      return "eevee_depth_of_field_scatter_foreground_lut";
    case DOF_SCATTER_FOREGROUND:
      return "eevee_depth_of_field_scatter_foreground";
    case DOF_SETUP:
      return "eevee_depth_of_field_setup";
    case DOF_TILES_DILATE_MINABS:
      return "eevee_depth_of_field_tiles_dilate_minabs";
    case DOF_TILES_DILATE_MINMAX:
      return "eevee_depth_of_field_tiles_dilate_minmax";
    case DOF_TILES_FLATTEN:
      return "eevee_depth_of_field_tiles_flatten";

    case LIGHTPROBE_DISPLAY_CUBEMAP:
      return "eevee_lightprobe_display_cubemap";
    case LIGHTPROBE_DISPLAY_IRRADIANCE:
      return "eevee_lightprobe_display_grid";
    case LIGHTPROBE_FILTER_DOWNSAMPLE_CUBE:
      return "eevee_lightprobe_filter_downsample";
    case LIGHTPROBE_FILTER_GLOSSY:
      return "eevee_lightprobe_filter_glossy";
    case LIGHTPROBE_FILTER_DIFFUSE:
      return "eevee_lightprobe_filter_diffuse";
    case LIGHTPROBE_FILTER_VISIBILITY:
      return "eevee_lightprobe_filter_visibility";

    case LOOKDEV_BACKGROUND:
      return "eevee_background_lookdev";

    case MOTION_BLUR_GATHER:
      return "eevee_motion_blur_gather";
    case MOTION_BLUR_TILE_DILATE:
      return "eevee_motion_blur_tiles_dilate";
    case MOTION_BLUR_TILE_FLATTEN:
      return "eevee_motion_blur_tiles_flatten";

    case RAYTRACE_DIFFUSE:
      return "eevee_raytrace_raygen_diffuse";
    case RAYTRACE_REFLECTION:
      return "eevee_raytrace_raygen_reflection";
    case RAYTRACE_REFRACTION:
      return "eevee_raytrace_raygen_refraction";
    case RAYTRACE_DIFFUSE_FALLBACK:
      return "eevee_raytrace_raygen_fallback_diffuse";
    case RAYTRACE_REFLECTION_FALLBACK:
      return "eevee_raytrace_raygen_fallback_reflection";
    case RAYTRACE_REFRACTION_FALLBACK:
      return "eevee_raytrace_raygen_fallback_refraction";
    case RAYTRACE_DENOISE_DIFFUSE:
      return "eevee_raytrace_denoise_diffuse";
    case RAYTRACE_DENOISE_REFLECTION:
      return "eevee_raytrace_denoise_reflection";
    case RAYTRACE_DENOISE_REFRACTION:
      return "eevee_raytrace_denoise_refraction";
    case RAYTRACE_RESOLVE_DIFFUSE:
      return "eevee_raytrace_resolve_diffuse";
    case RAYTRACE_RESOLVE_REFLECTION:
      return "eevee_raytrace_resolve_reflection";
    case RAYTRACE_RESOLVE_REFRACTION:
      return "eevee_raytrace_resolve_refraction";

    case SHADOW_DEBUG:
      return "eevee_shadow_debug";
    case SHADOW_PAGE_ALLOC:
      return "eevee_shadow_page_alloc";
    case SHADOW_PAGE_COPY:
      return "eevee_shadow_page_copy";
    case SHADOW_PAGE_DEBUG:
      return "eevee_shadow_page_debug";
    case SHADOW_PAGE_DEFRAG:
      return "eevee_shadow_page_defrag";
    case SHADOW_PAGE_FREE:
      return "eevee_shadow_page_free";
    case SHADOW_PAGE_INIT:
      return "eevee_shadow_page_init";
    case SHADOW_PAGE_MARK:
      return "eevee_shadow_page_mark";
    case SHADOW_TILE_DEPTH_SCAN:
      return "eevee_shadow_tilemap_depth_scan";
    case SHADOW_TILE_LOD_MASK:
      return "eevee_shadow_tilemap_lod_mask";
    case SHADOW_TILE_SETUP:
      return "eevee_shadow_tilemap_setup";
    case SHADOW_TILE_TAG_UPDATE:
      return "eevee_shadow_tilemap_tag_update";
    case SHADOW_TILE_TAG_USAGE:
      return "eevee_shadow_tilemap_tag_usage";
    case SHADOW_TILE_TAG_VISIBILITY:
      return "eevee_shadow_tilemap_visibility";

    case SUBSURFACE_EVAL:
      return "eevee_subsurface_eval";

    case VELOCITY_MESH:
      return "eevee_velocity_surface_mesh";
    case VELOCITY_CAMERA:
      return "eevee_velocity_camera";
    /* To avoid compiler warning about missing case. */
    case MAX_SHADER_TYPE:
      return "";
  }
  return "";
}

GPUShader *ShaderModule::static_shader_get(eShaderType shader_type)
{
  if (shaders_[shader_type] == nullptr) {
    ShaderDescription &desc = shader_descriptions_[shader_type];
    if (desc.compute_shader_code != nullptr) {
      const GPUShaderCreateInfo *create_info = static_shader_create_info_name_get(shader_type);
      char *comp_with_lib = DRW_shader_library_create_shader_string(shader_lib_, create_info);

      shaders_[shader_type] = GPU_shader_create_from_info_name(
          comp_with_lib, nullptr, desc.defines_shader_code, desc.name);

      MEM_SAFE_FREE(comp_with_lib);
    }
    else {
      shaders_[shader_type] = DRW_shader_create_with_shaderlib_ex(desc.vertex_shader_code,
                                                                  desc.geometry_shader_code,
                                                                  desc.fragment_shader_code,
                                                                  shader_lib_,
                                                                  desc.defines_shader_code,
                                                                  desc.name);
    }
    if (shaders_[shader_type] == nullptr) {
      fprintf(stderr, "EEVEE: error: Could not compile static shader \"%s\"\n", desc.name);
    }
    BLI_assert(shaders_[shader_type] != nullptr);
  }
  return shaders_[shader_type];
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name GPU Materials
 *
 * \{ */

char *ShaderModule::material_shader_code_defs_get(eMaterialGeometry geometry_type)
{
  std::string output = "";

  switch (geometry_type) {
    case MAT_GEOM_HAIR:
      output += "#define MAT_GEOM_HAIR\n";
      break;
    case MAT_GEOM_GPENCIL:
      output += "#define MAT_GEOM_GPENCIL\n";
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
    output += "#pragma BLENDER_REQUIRE(common_obinfos_lib.glsl)\n";
  }

  if (codegen->attribs_interface) {
    /* Declare inputs. */
    std::string delimiter = ";\n";
    std::string sub(codegen->attribs_declare);
    size_t start, pos = 0;
    while ((pos = sub.find(delimiter)) != std::string::npos) {
      switch (geometry_type) {
        case MAT_GEOM_MESH:
          /* Example print:
           * in float2 u015684; */
          output += "in ";
          output += sub.substr(0, pos + delimiter.length());
          break;
        case MAT_GEOM_HAIR:
          /* Example print:
           * uniform samplerBuffer u015684; */
          output += "uniform samplerBuffer ";
          start = sub.find(" ") + 1;
          output += sub.substr(start, pos + delimiter.length() - start);
          break;
        case MAT_GEOM_GPENCIL:
          /* Example print:
           * float2 u015684;
           * These are not used and just here to make the attribs_load functions call valids.
           * Only one uv and one color attribute layer is supported by gpencil objects. */
          output += sub.substr(0, pos + delimiter.length());
          break;
        case MAT_GEOM_WORLD:
        case MAT_GEOM_VOLUME:
        case MAT_GEOM_LOOKDEV:
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

  if (ELEM(geometry_type, MAT_GEOM_MESH, MAT_GEOM_HAIR)) {
    if (codegen->displacement) {
      if (GPU_material_flag_get(mat, GPU_MATFLAG_UNIFORMS_ATTRIB)) {
        output += datatoc_common_uniform_attribute_lib_glsl;
      }
      output += codegen->uniforms;
      output += "\n";
      output += codegen->library;
      output += "\n";
    }

    output += "float3 nodetree_displacement(void)\n";
    output += "{\n";
    if (codegen->displacement) {
      output += codegen->displacement;
    }
    else {
      output += "return float3(0);\n";
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
    case MAT_GEOM_LOOKDEV:
      output += datatoc_eevee_surface_lookdev_vert_glsl;
      break;
    case MAT_GEOM_HAIR:
      output += datatoc_eevee_surface_hair_vert_glsl;
      break;
    case MAT_GEOM_MESH:
    default:
      output += datatoc_eevee_surface_mesh_vert_glsl;
      break;
  }

  return DRW_shader_library_create_shader_string(shader_lib_, output.c_str());
}

char *ShaderModule::material_shader_code_geom_get(const GPUCodegenOutput *codegen,
                                                  GPUMaterial *mat,
                                                  eMaterialGeometry geometry_type)
{
  /* Force geometry usage if GPU_BARYCENTRIC_DIST is used. */
  if (!GPU_material_flag_get(mat, GPU_MATFLAG_BARYCENTRIC) ||
      !ELEM(geometry_type, MAT_GEOM_GPENCIL, MAT_GEOM_MESH)) {
    return nullptr;
  }

  StringRefNull interp_lib(datatoc_eevee_surface_lib_glsl);
  int64_t start = interp_lib.find("SurfaceInterface");
  int64_t end = interp_lib.find("interp");
  StringRef interp_lib_stripped = interp_lib.substr(start, end - start);
  std::string output = "\n\n";

  if (codegen->attribs_interface) {
    output += "in AttributesInterface\n";
    output += "{\n";
    output += codegen->attribs_interface;
    output += "} attr_in[];\n\n";

    output += "out AttributesInterface\n";
    output += "{\n";
    output += codegen->attribs_interface;
    output += "} attr_out;\n\n";
  }

  output += "in ";
  output += interp_lib_stripped;
  output += "interp_in[];\n\n";

  output += "out ";
  output += interp_lib_stripped;
  output += "interp_out;\n\n";

  output += datatoc_eevee_surface_mesh_geom_glsl;

  output += "void main(void)\n";
  output += "{\n";
  output += "interp_out.barycentric_dists = calc_barycentric_distances(";
  output += "  interp_in[0].P, interp_in[1].P, interp_in[2].P);\n ";

  for (int i = 0; i < 3; i++) {
    output += "{\n";
    output += "const int vert_id = " + std::to_string(i) + ";\n";
    output += "interp_out.barycentric_coords = calc_barycentric_co(vert_id);";
    output += "gl_Position = gl_in[vert_id].gl_Position;";
    if (codegen->attribs_passthrough) {
      output += codegen->attribs_passthrough;
    }
    output += "EmitVertex();";
    output += "}\n";
  }
  output += "}\n";

  return DRW_shader_library_create_shader_string(shader_lib_, output.c_str());
}

char *ShaderModule::material_shader_code_frag_get(const GPUCodegenOutput *codegen,
                                                  GPUMaterial *gpumat,
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
         * float2 u015684;
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
    if (GPU_material_flag_get(gpumat, GPU_MATFLAG_UNIFORMS_ATTRIB)) {
      output += datatoc_common_uniform_attribute_lib_glsl;
    }
    if (GPU_material_flag_get(gpumat, GPU_MATFLAG_OBJECT_INFO)) {
      output += "#pragma BLENDER_REQUIRE(common_obinfos_lib.glsl)\n";
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

  output += "float nodetree_thickness(void)\n";
  output += "{\n";
  if (codegen->thickness) {
    output += codegen->thickness;
  }
  else {
    /* TODO(fclem): Better default. */
    output += "return 0.1;\n";
  }
  output += "}\n\n";

  switch (geometry_type) {
    case MAT_GEOM_WORLD:
      output += datatoc_eevee_surface_background_frag_glsl;
      break;
    case MAT_GEOM_VOLUME:
      switch (pipeline_type) {
        case MAT_PIPE_DEFERRED:
          output += datatoc_eevee_volume_deferred_frag_glsl;
          break;
        default:
          BLI_assert(0);
          break;
      }
      break;
    default:
      switch (pipeline_type) {
        case MAT_PIPE_FORWARD_PREPASS:
          output += datatoc_eevee_surface_depth_simple_frag_glsl;
          break;
        case MAT_PIPE_DEFERRED_PREPASS:
        case MAT_PIPE_SHADOW:
          if (GPU_material_flag_get(gpumat, GPU_MATFLAG_TRANSPARENT)) {
            output += datatoc_eevee_surface_depth_frag_glsl;
          }
          else {
            output += datatoc_eevee_surface_depth_simple_frag_glsl;
          }
          break;
        case MAT_PIPE_DEFERRED:
          output += datatoc_eevee_surface_deferred_frag_glsl;
          break;
        case MAT_PIPE_FORWARD:
          output += datatoc_eevee_surface_forward_frag_glsl;
          break;
        default:
          BLI_assert(0);
          break;
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
  material_type_from_shader_uuid(shader_uuid, pipeline_type, geometry_type);

  GPUShaderSource source;
  source.vertex = material_shader_code_vert_get(codegen, mat, geometry_type);
  source.fragment = material_shader_code_frag_get(codegen, mat, geometry_type, pipeline_type);
  source.geometry = material_shader_code_geom_get(codegen, mat, geometry_type);
  source.defines = material_shader_code_defs_get(geometry_type);
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
                                               eMaterialPipeline pipeline_type,
                                               eMaterialGeometry geometry_type,
                                               bool deferred_compilation)
{
  uint64_t shader_uuid = shader_uuid_from_material_type(pipeline_type, geometry_type);

  bool is_volume = (pipeline_type == MAT_PIPE_VOLUME);

  return DRW_shader_from_material(
      blender_mat, nodetree, shader_uuid, is_volume, deferred_compilation, codegen_callback, this);
}

GPUMaterial *ShaderModule::world_shader_get(::World *blender_world, struct bNodeTree *nodetree)
{
  eMaterialPipeline pipeline_type = MAT_PIPE_DEFERRED; /* Unused. */
  eMaterialGeometry geometry_type = MAT_GEOM_WORLD;

  uint64_t shader_uuid = shader_uuid_from_material_type(pipeline_type, geometry_type);

  bool is_volume = (pipeline_type == MAT_PIPE_VOLUME);
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
                                               eMaterialPipeline pipeline_type,
                                               eMaterialGeometry geometry_type,
                                               bool is_lookdev)
{
  uint64_t shader_uuid = shader_uuid_from_material_type(pipeline_type, geometry_type);

  bool is_volume = (pipeline_type == MAT_PIPE_VOLUME);

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
