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
#include "GPU_material.h"
#include "GPU_shader.h"

#include "eevee_id_map.hh"

namespace blender::eevee {

/* Keep alphabetical order and clean prefix. */
enum eShaderType {
  CULLING_DEBUG = 0,
  CULLING_LIGHT,

  DEFERRED_EVAL_DIRECT,
  DEFERRED_EVAL_HOLDOUT,
  DEFERRED_EVAL_TRANSPARENT,
  DEFERRED_EVAL_VOLUME,

  DEFERRED_MESH,
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

  HIZ_COPY,
  HIZ_DOWNSAMPLE,

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

  RAYTRACE_DIFFUSE,
  RAYTRACE_DIFFUSE_FALLBACK,
  RAYTRACE_REFLECTION,
  RAYTRACE_REFLECTION_FALLBACK,
  RAYTRACE_REFRACTION,
  RAYTRACE_REFRACTION_FALLBACK,
  RAYTRACE_DENOISE_DIFFUSE,
  RAYTRACE_DENOISE_REFLECTION,
  RAYTRACE_DENOISE_REFRACTION,
  RAYTRACE_RESOLVE_DIFFUSE,
  RAYTRACE_RESOLVE_REFLECTION,
  RAYTRACE_RESOLVE_REFRACTION,

  SHADOW_DEBUG,
  SHADOW_PAGE_ALLOC,
  SHADOW_PAGE_COPY,
  SHADOW_PAGE_FREE,
  SHADOW_PAGE_INIT,
  SHADOW_PAGE_MARK,
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
  struct ShaderDescription {
    const char *name = nullptr;
    const char *vertex_shader_code = nullptr;
    const char *geometry_shader_code = nullptr;
    const char *fragment_shader_code = nullptr;
    const char *compute_shader_code = nullptr;
    const char *defines_shader_code = nullptr;
  };

  DRWShaderLibrary *shader_lib_ = nullptr;
  std::array<GPUShader *, MAX_SHADER_TYPE> shaders_;
  std::array<ShaderDescription, MAX_SHADER_TYPE> shader_descriptions_;
  std::string shared_lib_;

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

  GPUShaderSource material_shader_code_generate(GPUMaterial *mat, const GPUCodegenOutput *codegen);

 private:
  /* Run some custom preprocessor shader rewrite and returns a new string. */
  std::string enum_preprocess(const char *input);

  char *material_shader_code_defs_get(eMaterialGeometry geometry_type);
  char *material_shader_code_vert_get(const GPUCodegenOutput *codegen,
                                      GPUMaterial *mat,
                                      eMaterialGeometry geometry_type);
  char *material_shader_code_geom_get(const GPUCodegenOutput *codegen,
                                      GPUMaterial *mat,
                                      eMaterialGeometry geometry_type);
  char *material_shader_code_frag_get(const GPUCodegenOutput *codegen,
                                      GPUMaterial *mat,
                                      eMaterialGeometry geometry_type,
                                      eMaterialPipeline pipeline_type);
};

}  // namespace blender::eevee
