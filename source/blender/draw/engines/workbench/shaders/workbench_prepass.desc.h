
#include "gpu_shader_descriptor.h"

/* -------------------------------------------------------------------- */
/** \name Object Type
 * \{ */

GPU_SHADER_DESCRIPTOR(workbench_mesh,
                      {
                          .vertex_inputs =
                              {
                                  [0] = VERTEX_INPUT(VEC3, "pos"),
                                  [1] = VERTEX_INPUT(VEC3, "nor"),
                                  [2] = VERTEX_INPUT(VEC4, "ac"),
                                  [3] = VERTEX_INPUT(VEC2, "au"),
                              },
                          .vertex_source = "workbench_prepass_vert.glsl",
                      })

GPU_SHADER_DESCRIPTOR(workbench_hair,
                      {
                          .vertex_source = "workbench_prepass_hair_vert.glsl",
                          .resources =
                              {
                                  [DESCRIPTOR_SET_0] =
                                      {
                                          [0] = SAMPLER(FLOAT_BUFFER, "ac", GPU_SAMPLER_DEFAULT),
                                          [1] = SAMPLER(FLOAT_BUFFER, "au", GPU_SAMPLER_DEFAULT),
                                      },
                              },
                          .additional_descriptors =
                              {
                                  /* TODO */
                                  // &draw_hair,
                              },
                      })

GPU_SHADER_DESCRIPTOR(workbench_pointcloud,
                      {
                          .vertex_inputs =
                              {
                                  [0] = VERTEX_INPUT(VEC4, "pos"),
                                  /* ---- Instanced attribs ---- */
                                  [1] = VERTEX_INPUT(VEC3, "pos_inst"),
                                  [2] = VERTEX_INPUT(VEC3, "nor"),
                              },
                          .vertex_source = "workbench_prepass_pointcloud_vert.glsl",
                          .additional_descriptors =
                              {
                                  /* TODO */
                                  // &draw_pointcloud,
                              },
                      })

/** \} */

/* -------------------------------------------------------------------- */
/** \name Texture Type
 * \{ */

GPU_SHADER_DESCRIPTOR(workbench_texture_none,
                      {
                          .defines =
                              {
                                  [0] = "#define TEXTURE_NONE\n",
                              },
                      })

GPU_SHADER_DESCRIPTOR(
    workbench_texture_single,
    {
        .resources =
            {
                [DESCRIPTOR_SET_0] =
                    {
                        /* ---- Hair buffers ---- */
                        // [0] = SAMPLER(FLOAT_BUFFER, "ac", GPU_SAMPLER_DEFAULT),
                        // [1] = SAMPLER(FLOAT_BUFFER, "au", GPU_SAMPLER_DEFAULT),
                        [2] = SAMPLER(FLOAT_2D, "imageTexture", GPU_SAMPLER_DEFAULT),
                    },
            },
        .defines =
            {
                [0] = "#define V3D_SHADING_TEXTURE_COLOR\n",
            },
    })

GPU_SHADER_DESCRIPTOR(
    workbench_texture_tile,
    {
        .resources =
            {
                [DESCRIPTOR_SET_0] =
                    {
                        /* ---- Hair buffers ---- */
                        // [0] = SAMPLER(FLOAT_BUFFER, "ac", GPU_SAMPLER_DEFAULT),
                        // [1] = SAMPLER(FLOAT_BUFFER, "au", GPU_SAMPLER_DEFAULT),
                        [2] = SAMPLER(FLOAT_2D_ARRAY, "imageTileArray", GPU_SAMPLER_DEFAULT),
                        [3] = SAMPLER(FLOAT_1D_ARRAY, "imageTileData", GPU_SAMPLER_DEFAULT),
                    },
            },
        .defines =
            {
                [0] = "#define TEXTURE_IMAGE_ARRAY\n",
            },
    })

/** \} */

/* -------------------------------------------------------------------- */
/** \name Lighting Type
 * \{ */

GPU_SHADER_DESCRIPTOR(workbench_lighting_studio,
                      {
                          .defines =
                              {
                                  [1] = "#define V3D_LIGHTING_STUDIO\n",
                              },
                      })

GPU_SHADER_DESCRIPTOR(workbench_lighting_matcap,
                      {
                          .defines =
                              {
                                  [1] = "#define V3D_LIGHTING_MATCAP\n",
                              },
                      })

GPU_SHADER_DESCRIPTOR(workbench_lighting_flat,
                      {
                          .defines =
                              {
                                  [1] = "#define V3D_LIGHTING_FLAT\n",
                              },
                      })

/** \} */

/* -------------------------------------------------------------------- */
/** \name Material Interface
 * \{ */

GPU_STAGE_INTERFACE_CREATE(workbench_material_iface,
                           {
                               {VEC3, "normal_interp"},
                               {VEC3, "color_interp"},
                               {FLOAT, "alpha_interp"},
                               {VEC2, "uv_interp"},
                               {INT, "object_id", FLAT},
                               {FLOAT, "roughness", FLAT},
                               {FLOAT, "metallic", FLAT},
                           })

GPU_SHADER_DESCRIPTOR(workbench_material,
                      {
                          .vertex_out_interfaces =
                              {
                                  [0] = STAGE_INTERFACE("", workbench_material_iface),
                              },
                          .additional_descriptors =
                              {
                                  &draw_view,
                              },
                      })

/** \} */

/* -------------------------------------------------------------------- */
/** \name Pipeline Type
 * \{ */

GPU_SHADER_DESCRIPTOR(workbench_transparent_accum,
                      {
                          .fragment_outputs =
                              {
                                  /* Note: Blending will be skipped on objectId because output is a
                                     non-normalized integer buffer. */
                                  [0] = FRAGMENT_OUTPUT(VEC4, "transparentAccum"),
                                  [1] = FRAGMENT_OUTPUT(VEC4, "revealageAccum"),
                                  [2] = FRAGMENT_OUTPUT(UINT, "objectId"),
                              },
                          .fragment_source = "workbench_effect_cavity_frag.glsl",
                          .additional_descriptors =
                              {
                                  &workbench_material,
                              },
                      })

GPU_SHADER_DESCRIPTOR(workbench_opaque,
                      {
                          .fragment_outputs =
                              {
                                  /* Note: Blending will be skipped on objectId because output is a
                                     non-normalized integer buffer. */
                                  [0] = FRAGMENT_OUTPUT(VEC4, "materialData"),
                                  [1] = FRAGMENT_OUTPUT(VEC2, "normalData"),
                                  [2] = FRAGMENT_OUTPUT(UINT, "objectId"),
                              },
                          .resources =
                              {
                                  [DESCRIPTOR_SET_0] =
                                      {
                                          [4] = UNIFORM_BUFFER("WB_Scene", "scene"),
                                      },
                              },
                          .fragment_source = "workbench_effect_cavity_frag.glsl",
                          .additional_descriptors =
                              {
                                  &workbench_material,
                              },
                      })

/** \} */

/* -------------------------------------------------------------------- */
/** \name Variations Declaration
 * \{ */

#define WORKBENCH_SURFACETYPE_VARIATIONS(prefix, ...) \
  GPU_SHADER_DESCRIPTOR(prefix##_mesh, \
                        { \
                            .additional_descriptors = \
                                { \
                                    &workbench_mesh, \
                                    __VA_ARGS__, \
                                }, \
                        }) \
  GPU_SHADER_DESCRIPTOR(prefix##_hair, \
                        { \
                            .additional_descriptors = \
                                { \
                                    &workbench_hair, \
                                    __VA_ARGS__, \
                                }, \
                        }) \
  GPU_SHADER_DESCRIPTOR(prefix##_pointcloud, \
                        { \
                            .additional_descriptors = \
                                { \
                                    &workbench_pointcloud, \
                                    __VA_ARGS__, \
                                }, \
                        })

#define WORKBENCH_PIPELINE_VARIATIONS(prefix, ...) \
  WORKBENCH_SURFACETYPE_VARIATIONS(prefix##_transp_studio, \
                                   &workbench_transparent_accum, \
                                   &workbench_lighting_studio, \
                                   __VA_ARGS__) \
  WORKBENCH_SURFACETYPE_VARIATIONS(prefix##_transp_matcap, \
                                   &workbench_transparent_accum, \
                                   &workbench_lighting_matcap, \
                                   __VA_ARGS__) \
  WORKBENCH_SURFACETYPE_VARIATIONS( \
      prefix##_transp_flat, &workbench_transparent_accum, &workbench_lighting_flat, __VA_ARGS__) \
  WORKBENCH_SURFACETYPE_VARIATIONS(prefix##_opaque, &workbench_opaque, __VA_ARGS__)

WORKBENCH_PIPELINE_VARIATIONS(workbench_tex_none, &workbench_texture_none)
WORKBENCH_PIPELINE_VARIATIONS(workbench_tex_single, &workbench_texture_single)
WORKBENCH_PIPELINE_VARIATIONS(workbench_tex_tile, &workbench_texture_tile)

/** \} */
