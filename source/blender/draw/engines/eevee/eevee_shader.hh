/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2021 Blender Foundation.
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
#include "GPU_material.h"
#include "GPU_shader.h"

#include "eevee_id_map.hh"

namespace blender::eevee {

/* Keep alphabetical order and clean prefix. */
enum eShaderType {
  CULLING_DEBUG = 0,
  CULLING_SELECT,
  CULLING_SORT,
  CULLING_TILE,

  DEFERRED_EVAL,

  DEFERRED_VOLUME,

  DOF_BOKEH_LUT,
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
  FILM_RESOLVE_DEPTH,

  HIZ_UPDATE,

  LIGHTPROBE_DISPLAY_CUBEMAP,
  LIGHTPROBE_DISPLAY_IRRADIANCE,

  LIGHTPROBE_FILTER_DOWNSAMPLE_CUBE,
  LIGHTPROBE_FILTER_GLOSSY,
  LIGHTPROBE_FILTER_DIFFUSE,
  LIGHTPROBE_FILTER_VISIBILITY,

  LOOKDEV_BACKGROUND,

  MOTION_BLUR_GATHER,
  MOTION_BLUR_TILE_DILATE,
  MOTION_BLUR_TILE_FLATTEN,

  RAYTRACE_DISPATCH,
  RAYTRACE_RAYGEN,
  RAYTRACE_SCREEN_REFLECT,
  RAYTRACE_SCREEN_REFRACT,

  SHADOW_DEBUG,
  SHADOW_PAGE_ALLOC,
  SHADOW_PAGE_COPY,
  SHADOW_PAGE_DEBUG,
  SHADOW_PAGE_DEFRAG,
  SHADOW_PAGE_FREE,
  SHADOW_PAGE_INIT,
  SHADOW_PAGE_LIST,
  SHADOW_PAGE_MARK,
  SHADOW_TILE_DEPTH_SCAN,
  SHADOW_TILE_LOD_MASK,
  SHADOW_TILE_SETUP,
  SHADOW_TILE_TAG_UPDATE,
  SHADOW_TILE_TAG_USAGE,
  SHADOW_TILE_TAG_VISIBILITY,

  SUBSURFACE_EVAL,

  VELOCITY_CAMERA,
  VELOCITY_MESH,

  MAX_SHADER_TYPE,
};

/**
 * Shader module. shared between instances.
 */
class ShaderModule {
 private:
  std::array<GPUShader *, MAX_SHADER_TYPE> shaders_;

  /** Shared shader module accross all engine instances. */
  static ShaderModule *g_shader_module;

 public:
  ShaderModule();
  ~ShaderModule();

  GPUShader *static_shader_get(eShaderType shader_type);
  GPUMaterial *material_shader_get(::Material *blender_mat,
                                   struct bNodeTree *nodetree,
                                   eMaterialPipeline pipeline_type,
                                   eMaterialGeometry geometry_type,
                                   bool deferred_compilation);
  GPUMaterial *world_shader_get(::World *blender_world, struct bNodeTree *nodetree);
  GPUMaterial *material_shader_get(const char *name,
                                   ListBase &materials,
                                   struct bNodeTree *nodetree,
                                   eMaterialPipeline pipeline_type,
                                   eMaterialGeometry geometry_type,
                                   bool is_lookdev);

  void material_create_info_ammend(GPUMaterial *mat, GPUCodegenOutput *codegen);

  /** Only to be used by Instance constructor. */
  static ShaderModule *module_get();
  static void module_free();

 private:
  const char *static_shader_create_info_name_get(eShaderType shader_type);
};

}  // namespace blender::eevee
