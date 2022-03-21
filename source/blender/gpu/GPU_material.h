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
 * The Original Code is Copyright (C) 2005 Blender Foundation.
 * All rights reserved.
 */

/** \file
 * \ingroup gpu
 */

#pragma once

#include "DNA_customdata_types.h" /* for CustomDataType */
#include "DNA_image_types.h"
#include "DNA_listBase.h"
#include "DNA_scene_types.h"

#include "BLI_sys_types.h" /* for bool */

#include "GPU_shader.h"  /* for GPUShaderCreateInfo */
#include "GPU_texture.h" /* for eGPUSamplerState */

#ifdef __cplusplus
extern "C" {
#endif

struct GHash;
struct GPUMaterial;
struct GPUNode;
struct GPUNodeLink;
struct GPUNodeStack;
struct GPUTexture;
struct GPUUniformBuf;
struct Image;
struct ImageUser;
struct ListBase;
struct Main;
struct Material;
struct Scene;
struct bNode;
struct bNodeTree;

typedef struct GPUMaterial GPUMaterial;
typedef struct GPUNode GPUNode;
typedef struct GPUNodeLink GPUNodeLink;

/* Functions to create GPU Materials nodes. */

typedef enum eGPUType {
  /* Keep in sync with GPU_DATATYPE_STR */
  /* The value indicates the number of elements in each type */
  GPU_NONE = 0,
  GPU_FLOAT = 1,
  GPU_VEC2 = 2,
  GPU_VEC3 = 3,
  GPU_VEC4 = 4,
  GPU_MAT3 = 9,
  GPU_MAT4 = 16,
  GPU_MAX_CONSTANT_DATA = GPU_MAT4,

  /* Values not in GPU_DATATYPE_STR */
  GPU_TEX1D_ARRAY = 1001,
  GPU_TEX2D = 1002,
  GPU_TEX2D_ARRAY = 1003,
  GPU_TEX3D = 1004,
  GPU_IMAGE_2D = 1005,

  /* GLSL Struct types */
  GPU_CLOSURE = 1007,

  /* Opengl Attributes */
  GPU_ATTR = 3001,
} eGPUType;

typedef enum eGPUMaterialFlag {
  GPU_MATFLAG_DIFFUSE = (1 << 0),
  GPU_MATFLAG_SUBSURFACE = (1 << 1),
  GPU_MATFLAG_GLOSSY = (1 << 2),
  GPU_MATFLAG_REFRACT = (1 << 3),
  GPU_MATFLAG_EMISSION = (1 << 4),
  GPU_MATFLAG_TRANSPARENT = (1 << 5),
  GPU_MATFLAG_HOLDOUT = (1 << 6),
  GPU_MATFLAG_SHADER_TO_RGBA = (1 << 7),

  GPU_MATFLAG_OBJECT_INFO = (1 << 10),

  GPU_MATFLAG_BARYCENTRIC = (1 << 20),

  /* Tells the render engine the material was just compiled or updated. */
  GPU_MATFLAG_UPDATED = (1 << 29),

  /* HACK(fclem) Tells the environment texture node to not bail out if empty. */
  GPU_MATFLAG_LOOKDEV_HACK = (1 << 30),
} eGPUMaterialFlag;

ENUM_OPERATORS(eGPUMaterialFlag, GPU_MATFLAG_LOOKDEV_HACK);

typedef struct GPUNodeStack {
  eGPUType type;
  float vec[4];
  struct GPUNodeLink *link;
  bool hasinput;
  bool hasoutput;
  short sockettype;
  bool end;
} GPUNodeStack;

typedef enum eGPUMaterialStatus {
  GPU_MAT_FAILED = 0,
  GPU_MAT_CREATED,
  GPU_MAT_QUEUED,
  GPU_MAT_SUCCESS,
} eGPUMaterialStatus;

typedef enum eGPUVolumeDefaultValue {
  GPU_VOLUME_DEFAULT_0,
  GPU_VOLUME_DEFAULT_1,
} eGPUVolumeDefaultValue;

typedef struct GPUCodegenOutput {
  char *attr_load;
  /* Nodetree functions calls. */
  char *displacement;
  char *surface;
  char *volume;
  char *thickness;
  char *compute;

  GPUShaderCreateInfo *create_info;
} GPUCodegenOutput;

typedef void (*GPUCodegenCallbackFn)(void *thunk, GPUMaterial *mat, GPUCodegenOutput *codegen);

GPUNodeLink *GPU_constant(const float *num);
GPUNodeLink *GPU_uniform(const float *num);
GPUNodeLink *GPU_attribute(GPUMaterial *mat, CustomDataType type, const char *name);
GPUNodeLink *GPU_uniform_attribute(GPUMaterial *mat, const char *name, bool use_dupli);
GPUNodeLink *GPU_texture(GPUMaterial *material, eGPUSamplerState sampler_state);
GPUNodeLink *GPU_image(GPUMaterial *mat,
                       struct Image *ima,
                       struct ImageUser *iuser,
                       eGPUSamplerState sampler_state);
GPUNodeLink *GPU_image_tiled(GPUMaterial *mat,
                             struct Image *ima,
                             struct ImageUser *iuser,
                             eGPUSamplerState sampler_state);
GPUNodeLink *GPU_image_tiled_mapping(GPUMaterial *mat, struct Image *ima, struct ImageUser *iuser);
GPUNodeLink *GPU_color_band(GPUMaterial *mat, int size, float *pixels, float *row);
GPUNodeLink *GPU_volume_grid(GPUMaterial *mat,
                             const char *name,
                             eGPUVolumeDefaultValue default_value);
GPUNodeLink *GPU_image_texture(GPUMaterial *material, eGPUTextureFormat format);

bool GPU_link(GPUMaterial *mat, const char *name, ...);
bool GPU_stack_link(GPUMaterial *mat,
                    struct bNode *node,
                    const char *name,
                    GPUNodeStack *in,
                    GPUNodeStack *out,
                    ...);
/**
 * This is a special function to call the "*_eval" function of a BSDF node.
 * \note This must be call right after GPU_stack_link() so that out[0] contains a valid link.
 */
bool GPU_stack_eval_link(GPUMaterial *material,
                         struct bNode *bnode,
                         const char *name,
                         GPUNodeStack *in,
                         GPUNodeStack *out,
                         ...);
GPUNodeLink *GPU_uniformbuf_link_out(struct GPUMaterial *mat,
                                     struct bNode *node,
                                     struct GPUNodeStack *stack,
                                     int index);

void GPU_material_output_surface(GPUMaterial *material, GPUNodeLink *link);
void GPU_material_output_volume(GPUMaterial *material, GPUNodeLink *link);
void GPU_material_output_displacement(GPUMaterial *material, GPUNodeLink *link);
void GPU_material_output_thickness(GPUMaterial *material, GPUNodeLink *link);

void GPU_material_add_output_link_aov(GPUMaterial *material, GPUNodeLink *link, int hash);

void GPU_material_add_closure_eval(GPUMaterial *material,
                                   const GPUNodeLink *weight_link,
                                   const GPUNodeLink *eval_link);

/**
 * High level functions to create and use GPU materials.
 */
GPUMaterial *GPU_material_from_nodetree_find(struct ListBase *gpumaterials,
                                             const void *engine_type,
                                             int options);
/**
 * \note Caller must use #GPU_material_from_nodetree_find to re-use existing materials,
 * This is enforced since constructing other arguments to this function may be expensive
 * so only do this when they are needed.
 */
GPUMaterial *GPU_material_from_nodetree(struct Scene *scene,
                                        struct Material *ma,
                                        struct bNodeTree *ntree,
                                        struct ListBase *gpumaterials,
                                        const char *name,
                                        uint64_t shader_uuid,
                                        bool is_volume_shader,
                                        bool is_lookdev,
                                        GPUCodegenCallbackFn callback,
                                        void *thunk);

void GPU_material_compile(GPUMaterial *mat);
void GPU_material_free_single(GPUMaterial *material);
void GPU_material_free(struct ListBase *gpumaterial);

void GPU_materials_free(struct Main *bmain);

struct Scene *GPU_material_scene(GPUMaterial *material);
struct GPUPass *GPU_material_get_pass(GPUMaterial *material);
struct GPUShader *GPU_material_get_shader(GPUMaterial *material);
/**
 * Return can be NULL if it's a world material.
 */
struct Material *GPU_material_get_material(GPUMaterial *material);
/**
 * Return true if the material compilation has not yet begin or begin.
 */
eGPUMaterialStatus GPU_material_status(GPUMaterial *mat);
void GPU_material_status_set(GPUMaterial *mat, eGPUMaterialStatus status);

bool GPU_material_is_compute(GPUMaterial *material);
void GPU_material_is_compute_set(GPUMaterial *material, bool is_compute);

struct GPUUniformBuf *GPU_material_uniform_buffer_get(GPUMaterial *material);
/**
 * Create dynamic UBO from parameters
 *
 * \param inputs: Items are #LinkData, data is #GPUInput (`BLI_genericNodeN(GPUInput)`).
 */
void GPU_material_uniform_buffer_create(GPUMaterial *material, ListBase *inputs);
struct GPUUniformBuf *GPU_material_create_sss_profile_ubo(void);

bool GPU_material_is_volume_shader(GPUMaterial *mat);

void GPU_material_flag_set(GPUMaterial *mat, eGPUMaterialFlag flag);
bool GPU_material_flag_get(const GPUMaterial *mat, eGPUMaterialFlag flag);
bool GPU_material_recalc_flag_get(GPUMaterial *mat);
uint64_t GPU_material_uuid_get(GPUMaterial *mat);

void GPU_pass_cache_init(void);
void GPU_pass_cache_garbage_collect(void);
void GPU_pass_cache_free(void);

/* Requested Material Attributes and Textures */

typedef struct GPUMaterialAttribute {
  struct GPUMaterialAttribute *next, *prev;
  int type;      /* CustomDataType */
  char name[64]; /* MAX_CUSTOMDATA_LAYER_NAME */
  eGPUType gputype;
  int id;
  int users;
} GPUMaterialAttribute;

typedef struct GPUMaterialTexture {
  struct GPUMaterialTexture *next, *prev;
  struct Image *ima;
  struct ImageUser iuser;
  bool iuser_available;
  struct GPUTexture **colorband;
  char sampler_name[32];       /* Name of sampler in GLSL. */
  char tiled_mapping_name[32]; /* Name of tile mapping sampler in GLSL. */
  int users;
  int sampler_state; /* eGPUSamplerState */
} GPUMaterialTexture;

typedef struct GPUMaterialVolumeGrid {
  struct GPUMaterialVolumeGrid *next, *prev;
  char *name;
  eGPUVolumeDefaultValue default_value;
  char sampler_name[32];   /* Name of sampler in GLSL. */
  char transform_name[32]; /* Name of 4x4 matrix in GLSL. */
  int users;
} GPUMaterialVolumeGrid;

/* A reference to a write only 2D image of a specific format. */
typedef struct GPUMaterialImage {
  struct GPUMaterialImage *next, *prev;
  eGPUTextureFormat format;
  char name_in_shader[32];
} GPUMaterialImage;

ListBase GPU_material_attributes(GPUMaterial *material);
ListBase GPU_material_textures(GPUMaterial *material);
ListBase GPU_material_volume_grids(GPUMaterial *material);
ListBase GPU_material_images(GPUMaterial *material);

GPUMaterialTexture *GPU_material_get_link_texture(GPUNodeLink *link);
GPUMaterialImage *GPU_material_get_link_image(GPUNodeLink *link);

typedef struct GPUUniformAttr {
  struct GPUUniformAttr *next, *prev;

  /* Meaningful part of the attribute set key. */
  char name[64]; /* MAX_CUSTOMDATA_LAYER_NAME */
  bool use_dupli;

  /* Helper fields used by code generation. */
  short id;
  int users;
} GPUUniformAttr;

typedef struct GPUUniformAttrList {
  ListBase list; /* GPUUniformAttr */

  /* List length and hash code precomputed for fast lookup and comparison. */
  unsigned int count, hash_code;
} GPUUniformAttrList;

GPUUniformAttrList *GPU_material_uniform_attributes(GPUMaterial *material);

struct GHash *GPU_uniform_attr_list_hash_new(const char *info);
void GPU_uniform_attr_list_copy(GPUUniformAttrList *dest, GPUUniformAttrList *src);
void GPU_uniform_attr_list_free(GPUUniformAttrList *set);

typedef void (*GPUMaterialSetupFn)(void *thunk, GPUMaterial *material);
typedef void (*GPUMaterialCompileFn)(void *thunk, GPUMaterial *material);

/* Construct and immediately compile a GPU material from a set of callbacks. The setup callback
 * should set the appropriate flags or members to the material. The compile callback should
 * construct the material graph by adding and linking the necessary GPU material graph nodes. The
 * generate function should construct the needed shader by initializing the passed shader create
 * info structure. The given thunk will be passed as the first parameter of each callback. */
GPUMaterial *GPU_material_from_callbacks(GPUMaterialSetupFn setup_function,
                                         GPUMaterialCompileFn compile_function,
                                         GPUCodegenCallbackFn generate_function,
                                         void *thunk);

#ifdef __cplusplus
}
#endif
