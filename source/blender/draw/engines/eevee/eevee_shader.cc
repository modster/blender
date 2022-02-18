/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2021 Blender Foundation.
 */

/** \file
 * \ingroup eevee
 *
 * Shader module that manage shader libraries, deferred compilation,
 * and static shader usage.
 */

#include "eevee_shader.hh"
#include "eevee_material.hh"

#include "../../gpu/intern/gpu_shader_create_info.hh"

namespace blender::eevee {

/* -------------------------------------------------------------------- */
/** \name Module
 *
 * \{ */

ShaderModule *ShaderModule::g_shader_module = nullptr;

ShaderModule *ShaderModule::module_get()
{
  if (g_shader_module == nullptr) {
    /* TODO(fclem) threadsafety. */
    g_shader_module = new ShaderModule();
  }
  return g_shader_module;
}

void ShaderModule::module_free()
{
  if (g_shader_module != nullptr) {
    /* TODO(fclem) threadsafety. */
    delete g_shader_module;
    g_shader_module = nullptr;
  }
}

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

/** \} */

/* -------------------------------------------------------------------- */
/** \name Static shaders
 *
 * \{ */

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
      return "eevee_depth_of_field_resolve_lq_lut";
    case DOF_RESOLVE_LUT_HQ:
      return "eevee_depth_of_field_resolve_hq_lut";
    case DOF_RESOLVE:
      return "eevee_depth_of_field_resolve_lq";
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
    case SHADOW_PAGE_LIST:
      return "eevee_shadow_page_list";
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
    const char *shader_name = static_shader_create_info_name_get(shader_type);

    shaders_[shader_type] = GPU_shader_create_from_info_name(shader_name);

    if (shaders_[shader_type] == nullptr) {
      fprintf(stderr, "EEVEE: error: Could not compile static shader \"%s\"\n", shader_name);
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

void ShaderModule::material_create_info_ammend(GPUMaterial *gpumat, GPUCodegenOutput *codegen_)
{
  using namespace blender::gpu::shader;

  uint64_t shader_uuid = GPU_material_uuid_get(gpumat);

  eMaterialPipeline pipeline_type;
  eMaterialGeometry geometry_type;
  material_type_from_shader_uuid(shader_uuid, pipeline_type, geometry_type);

  GPUCodegenOutput &codegen = *codegen_;
  ShaderCreateInfo &info = *reinterpret_cast<ShaderCreateInfo *>(codegen.create_info);

  info.auto_resource_location(true);

  std::stringstream global_vars;
  switch (geometry_type) {
    case MAT_GEOM_MESH:
      /** Noop. */
      break;
    case MAT_GEOM_HAIR:
      /** Hair attributes comme from sampler buffer. Transfer attributes to sampler. */
      for (auto &input : info.vertex_inputs_) {
        info.sampler(0, ImageType::FLOAT_BUFFER, input.name, Frequency::BATCH);
      }
      info.vertex_inputs_.clear();
      break;
    case MAT_GEOM_WORLD:
      /**
       * Only orco layer is supported by world and it is procedurally generated. These are here to
       * make the attribs_load function calls valids.
       */
      ATTR_FALLTHROUGH;
    case MAT_GEOM_GPENCIL:
      /**
       * Only one uv and one color attribute layer are supported by gpencil objects and they are
       * already declared in another createInfo. These are here to make the attribs_load
       * function calls valids.
       */
      for (auto &input : info.vertex_inputs_) {
        global_vars << input.type << " " << input.name << ";\n";
      }
      info.vertex_inputs_.clear();
      break;
    case MAT_GEOM_VOLUME:
    case MAT_GEOM_LOOKDEV:
      /** No attributes supported. */
      info.vertex_inputs_.clear();
      break;
  }

  const bool do_fragment_attrib_load = (geometry_type == MAT_GEOM_WORLD);

  if (do_fragment_attrib_load && !info.vertex_out_interfaces_.is_empty()) {
    /* Codegen outputs only one interface. */
    const StageInterfaceInfo &iface = *info.vertex_out_interfaces_.first();
    /* Globals the attrib_load() can write to when it is in the fragment shader. */
    global_vars << "struct " << iface.name << " {\n";
    for (auto &inout : iface.inouts) {
      global_vars << "  " << inout.type << " " << inout.name << ";\n";
    }
    global_vars << "};\n";
    global_vars << iface.name << " " << iface.instance_name << ";\n";

    info.vertex_out_interfaces_.clear();
  }

  std::stringstream attr_load;
  attr_load << "void attrib_load()\n";
  attr_load << "{\n";
  attr_load << ((codegen.attr_load) ? codegen.attr_load : "");
  attr_load << "}\n\n";

  std::stringstream vert_gen, frag_gen;

  if (do_fragment_attrib_load) {
    frag_gen << global_vars.str() << attr_load.str();
  }
  else {
    vert_gen << global_vars.str() << attr_load.str();
  }

  {
    /* Only mesh and hair support displacement for now. */
    if (ELEM(geometry_type, MAT_GEOM_MESH, MAT_GEOM_HAIR)) {
      vert_gen << "vec3 nodetree_displacement()\n";
      vert_gen << "{\n";
      vert_gen << ((codegen.displacement) ? codegen.displacement : "return vec3(0);\n");
      vert_gen << "}\n\n";
    }

    info.vertex_source_generated = vert_gen.str();
  }

  {
    frag_gen << "Closure nodetree_surface()\n";
    frag_gen << "{\n";
    frag_gen << ((codegen.surface) ? codegen.surface : "return CLOSURE_DEFAULT;\n");
    frag_gen << "}\n\n";

    frag_gen << "Closure nodetree_volume()\n";
    frag_gen << "{\n";
    frag_gen << ((codegen.volume) ? codegen.volume : "return CLOSURE_DEFAULT;\n");
    frag_gen << "}\n\n";

    frag_gen << "float nodetree_thickness()\n";
    frag_gen << "{\n";
    /* TODO(fclem): Better default. */
    frag_gen << ((codegen.thickness) ? codegen.thickness : "return 0.1;\n");
    frag_gen << "}\n\n";

    info.fragment_source_generated = frag_gen.str();
  }

  /* Geometry Info. */
  switch (geometry_type) {
    case MAT_GEOM_WORLD:
      info.additional_info("eevee_surface_world");
      break;
    case MAT_GEOM_VOLUME:
      info.additional_info("eevee_volume");
      break;
    case MAT_GEOM_GPENCIL:
      info.additional_info("eevee_surface_gpencil");
      break;
    case MAT_GEOM_LOOKDEV:
      info.additional_info("eevee_surface_lookdev");
      break;
    case MAT_GEOM_HAIR:
      info.additional_info("eevee_surface_hair");
      break;
    case MAT_GEOM_MESH:
    default:
      info.additional_info("eevee_surface_mesh");
      break;
  }

  /* Pipeline Info. */
  switch (geometry_type) {
    case MAT_GEOM_WORLD:
      info.additional_info("eevee_surface_background");
      break;
    case MAT_GEOM_VOLUME:
      switch (pipeline_type) {
        case MAT_PIPE_DEFERRED:
          info.additional_info("eevee_volume_deferred");
          break;
        default:
          BLI_assert(0);
          break;
      }
      break;
    default:
      switch (pipeline_type) {
        case MAT_PIPE_FORWARD_PREPASS:
          info.additional_info("eevee_surface_depth_simple");
          break;
        case MAT_PIPE_DEFERRED_PREPASS:
        case MAT_PIPE_SHADOW:
          if (GPU_material_flag_get(gpumat, GPU_MATFLAG_TRANSPARENT)) {
            info.additional_info("eevee_surface_depth");
          }
          else {
            info.additional_info("eevee_surface_depth_simple");
            info.fragment_source_generated = "";
          }
          break;
        case MAT_PIPE_DEFERRED:
          info.additional_info("eevee_surface_deferred");
          break;
        case MAT_PIPE_FORWARD:
          info.additional_info("eevee_surface_forward");
          break;
        default:
          BLI_assert(0);
          break;
      }
      break;
  }
}

/* WATCH: This can be called from another thread! Needs to not touch the shader module in any
 * thread unsafe manner. */
static void codegen_callback(void *thunk, GPUMaterial *mat, GPUCodegenOutput *codegen)
{
  reinterpret_cast<ShaderModule *>(thunk)->material_create_info_ammend(mat, codegen);
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
