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
 * The Original Code is Copyright (C) 2006 Blender Foundation.
 * All rights reserved.
 */

/** \file
 * \ingroup gpu
 *
 * Manages materials, lights and textures.
 */

#include <math.h>
#include <string.h>

#include "MEM_guardedalloc.h"

#include "DNA_material_types.h"
#include "DNA_scene_types.h"
#include "DNA_world_types.h"

#include "BLI_listbase.h"
#include "BLI_math.h"
#include "BLI_string.h"
#include "BLI_string_utils.h"
#include "BLI_utildefines.h"

#include "BKE_main.h"
#include "BKE_material.h"
#include "BKE_node.h"
#include "BKE_scene.h"
#include "BKE_world.h"

#include "GPU_material.h"
#include "GPU_shader.h"
#include "GPU_texture.h"
#include "GPU_uniform_buffer.h"

#include "DRW_engine.h"

#include "gpu_codegen.h"
#include "gpu_node_graph.h"

/* Structs */
#define MAX_COLOR_BAND 128

typedef struct GPUColorBandBuilder {
  float pixels[MAX_COLOR_BAND][CM_TABLE + 1][4];
  int current_layer;
} GPUColorBandBuilder;

struct GPUMaterial {
  /* Contains GPUShader and source code for deferred compilation.
   * Can be shared between similar material (i.e: sharing same nodetree topology). */
  GPUPass *pass;
  /** UBOs for this material parameters. */
  GPUUniformBuf *ubo;
  /** Compilation status. Do not use if shader is not GPU_MAT_SUCCESS. */
  eGPUMaterialStatus status;
  /** Some flags about the nodetree & the needed resources. */
  eGPUMaterialFlag flag;
  /* Identify shader variations (shadow, probe, world background...).
   * Should be unique even across render engines. */
  uint64_t uuid;
  /** Object type for attribute fetching. */
  bool is_volume_shader;

  /** DEPRECATED Currently only used for deferred compilation. */
  Scene *scene;
  /** Source material, might be null. */
  Material *ma;
  /** 1D Texture array containing all color bands. */
  GPUTexture *coba_tex;
  GPUColorBandBuilder *coba_builder;
  /* Low level node graph(s). Also contains resources needed by the material. */
  GPUNodeGraph graph;

#ifndef NDEBUG
  char name[64];
#endif
};

enum {
  GPU_USE_SURFACE_OUTPUT = (1 << 0),
  GPU_USE_VOLUME_OUTPUT = (1 << 1),
};

/* Functions */

/* Returns the address of the future pointer to coba_tex */
GPUTexture **gpu_material_ramp_texture_row_set(GPUMaterial *mat,
                                               int size,
                                               float *pixels,
                                               float *row)
{
  /* In order to put all the color-bands into one 1D array texture,
   * we need them to be the same size. */
  BLI_assert(size == CM_TABLE + 1);
  UNUSED_VARS_NDEBUG(size);

  if (mat->coba_builder == NULL) {
    mat->coba_builder = MEM_mallocN(sizeof(GPUColorBandBuilder), "GPUColorBandBuilder");
    mat->coba_builder->current_layer = 0;
  }

  int layer = mat->coba_builder->current_layer;
  *row = (float)layer;

  if (*row == MAX_COLOR_BAND) {
    printf("Too many color band in shader! Remove some Curve, Black Body or Color Ramp Node.\n");
  }
  else {
    float *dst = (float *)mat->coba_builder->pixels[layer];
    memcpy(dst, pixels, sizeof(float) * (CM_TABLE + 1) * 4);
    mat->coba_builder->current_layer += 1;
  }

  return &mat->coba_tex;
}

static void gpu_material_ramp_texture_build(GPUMaterial *mat)
{
  if (mat->coba_builder == NULL) {
    return;
  }

  GPUColorBandBuilder *builder = mat->coba_builder;

  mat->coba_tex = GPU_texture_create_1d_array(
      "mat_ramp", CM_TABLE + 1, builder->current_layer, 1, GPU_RGBA16F, (float *)builder->pixels);

  MEM_freeN(builder);
  mat->coba_builder = NULL;
}

static void gpu_material_free_single(GPUMaterial *material)
{
  /* Cancel / wait any pending lazy compilation. */
  DRW_deferred_shader_remove(material);

  gpu_node_graph_free(&material->graph);

  if (material->pass != NULL) {
    GPU_pass_release(material->pass);
  }
  if (material->ubo != NULL) {
    GPU_uniformbuf_free(material->ubo);
  }
  if (material->coba_tex != NULL) {
    GPU_texture_free(material->coba_tex);
  }
}

void GPU_material_free(ListBase *gpumaterial)
{
  LISTBASE_FOREACH (LinkData *, link, gpumaterial) {
    GPUMaterial *material = link->data;
    gpu_material_free_single(material);
    MEM_freeN(material);
  }
  BLI_freelistN(gpumaterial);
}

Scene *GPU_material_scene(GPUMaterial *material)
{
  return material->scene;
}

GPUPass *GPU_material_get_pass(GPUMaterial *material)
{
  return material->pass;
}

GPUShader *GPU_material_get_shader(GPUMaterial *material)
{
  return material->pass ? GPU_pass_shader_get(material->pass) : NULL;
}

/* Return can be NULL if it's a world material. */
Material *GPU_material_get_material(GPUMaterial *material)
{
  return material->ma;
}

GPUUniformBuf *GPU_material_uniform_buffer_get(GPUMaterial *material)
{
  return material->ubo;
}

/**
 * Create dynamic UBO from parameters
 *
 * \param inputs: Items are #LinkData, data is #GPUInput (`BLI_genericNodeN(GPUInput)`).
 */
void GPU_material_uniform_buffer_create(GPUMaterial *material, ListBase *inputs)
{
#ifndef NDEBUG
  const char *name = material->name;
#else
  const char *name = "Material";
#endif
  material->ubo = GPU_uniformbuf_create_from_list(inputs, name);
}

ListBase GPU_material_attributes(GPUMaterial *material)
{
  return material->graph.attributes;
}

ListBase GPU_material_textures(GPUMaterial *material)
{
  return material->graph.textures;
}

ListBase GPU_material_volume_grids(GPUMaterial *material)
{
  return material->graph.volume_grids;
}

GPUUniformAttrList *GPU_material_uniform_attributes(GPUMaterial *material)
{
  GPUUniformAttrList *attrs = &material->graph.uniform_attrs;
  return attrs->count > 0 ? attrs : NULL;
}

void GPU_material_output_surface(GPUMaterial *material, GPUNodeLink *link)
{
  if (!material->graph.outlink_surface) {
    material->graph.outlink_surface = link;
  }
}

void GPU_material_output_volume(GPUMaterial *material, GPUNodeLink *link)
{
  if (!material->graph.outlink_volume) {
    material->graph.outlink_volume = link;
  }
}

void GPU_material_output_displacement(GPUMaterial *material, GPUNodeLink *link)
{
  if (!material->graph.outlink_displacement) {
    material->graph.outlink_displacement = link;
  }
}

void GPU_material_add_output_link_aov(GPUMaterial *material, GPUNodeLink *link, int hash)
{
  GPUNodeGraphOutputLink *aov_link = MEM_callocN(sizeof(GPUNodeGraphOutputLink), __func__);
  aov_link->outlink = link;
  aov_link->hash = hash;
  BLI_addtail(&material->graph.outlink_aovs, aov_link);
}

void GPU_material_add_closure_eval(GPUMaterial *material,
                                   const GPUNodeLink *weight_link,
                                   const GPUNodeLink *eval_link)
{
  GPUNodeGraphEvalNode *node = MEM_callocN(sizeof(GPUNodeGraphEvalNode), __func__);
  node->weight_node = weight_link->output->node;
  node->eval_node = eval_link->output->node;
  BLI_addtail(&material->graph.eval_nodes, node);
}

GPUNodeGraph *gpu_material_node_graph(GPUMaterial *material)
{
  return &material->graph;
}

eGPUMaterialStatus GPU_material_status(GPUMaterial *mat)
{
  return mat->status;
}

void GPU_material_status_set(GPUMaterial *mat, eGPUMaterialStatus status)
{
  mat->status = status;
}

/* Code generation */

bool GPU_material_is_volume_shader(GPUMaterial *mat)
{
  return mat->is_volume_shader;
}

void GPU_material_flag_set(GPUMaterial *mat, eGPUMaterialFlag flag)
{
  mat->flag |= flag;
}

bool GPU_material_flag_get(const GPUMaterial *mat, eGPUMaterialFlag flag)
{
  return (mat->flag & flag) != 0;
}

/* Note: Consumes the flags. */
bool GPU_material_recalc_flag_get(GPUMaterial *mat)
{
  bool updated = (mat->flag & GPU_MATFLAG_UPDATED) != 0;
  mat->flag &= ~GPU_MATFLAG_UPDATED;
  return updated;
}

uint64_t GPU_material_uuid_get(GPUMaterial *mat)
{
  return mat->uuid;
}

GPUMaterial *GPU_material_from_nodetree(Scene *scene,
                                        Material *ma,
                                        bNodeTree *ntree,
                                        ListBase *gpumaterials,
                                        const char *name,
                                        const uint64_t shader_uuid,
                                        const bool is_volume_shader,
                                        const bool is_lookdev,
                                        GPUCodegenCallbackFn callback,
                                        void *thunk)
{
  /* Search if this material is not already compiled. */
  LISTBASE_FOREACH (LinkData *, link, gpumaterials) {
    GPUMaterial *mat = (GPUMaterial *)link->data;
    if (mat->uuid == shader_uuid) {
      return mat;
    }
  }

  GPUMaterial *mat = MEM_callocN(sizeof(GPUMaterial), "GPUMaterial");
  mat->ma = ma;
  mat->scene = scene;
  mat->uuid = shader_uuid;
  mat->flag = GPU_MATFLAG_UPDATED;
  mat->is_volume_shader = is_volume_shader;
  mat->graph.used_libraries = BLI_gset_new(
      BLI_ghashutil_ptrhash, BLI_ghashutil_ptrcmp, "GPUNodeGraph.used_libraries");
#ifndef NDEBUG
  BLI_snprintf(mat->name, sizeof(mat->name), "%s", name);
#else
  UNUSED_VARS(name);
#endif
  if (is_lookdev) {
    mat->flag |= GPU_MATFLAG_LOOKDEV_HACK;
  }

  /* Localize tree to create links for reroute and mute. */
  bNodeTree *localtree = ntreeLocalize(ntree);
  ntreeGPUMaterialNodes(localtree, mat);

  gpu_material_ramp_texture_build(mat);

  if (mat->graph.outlink_surface || mat->graph.outlink_volume) {
    /* Create source code and search pass cache for an already compiled version. */
    mat->pass = GPU_generate_pass(mat, &mat->graph, callback, thunk);

    if (mat->pass == NULL) {
      /* We had a cache hit and the shader has already failed to compile. */
      mat->status = GPU_MAT_FAILED;
      gpu_node_graph_free(&mat->graph);
    }
    else {
      GPUShader *sh = GPU_pass_shader_get(mat->pass);
      if (sh != NULL) {
        /* We had a cache hit and the shader is already compiled. */
        mat->status = GPU_MAT_SUCCESS;
        gpu_node_graph_free_nodes(&mat->graph);
      }
      else {
        mat->status = GPU_MAT_CREATED;
      }
    }
  }
  else {
    mat->status = GPU_MAT_FAILED;
    gpu_node_graph_free(&mat->graph);
  }

  /* Only free after GPU_pass_shader_get where GPUUniformBuf read data from the local tree. */
  ntreeFreeLocalTree(localtree);
  BLI_assert(!localtree->id.py_instance); /* Or call #BKE_libblock_free_data_py. */
  MEM_freeN(localtree);

  /* Note that even if building the shader fails in some way, we still keep
   * it to avoid trying to compile again and again, and simply do not use
   * the actual shader on drawing. */
  LinkData *link = MEM_callocN(sizeof(LinkData), "GPUMaterialLink");
  link->data = mat;
  BLI_addtail(gpumaterials, link);

  return mat;
}

void GPU_material_compile(GPUMaterial *mat)
{
  bool success;

  BLI_assert(mat->status == GPU_MAT_QUEUED);
  BLI_assert(mat->pass);

  /* NOTE: The shader may have already been compiled here since we are
   * sharing GPUShader across GPUMaterials. In this case it's a no-op. */
#ifndef NDEBUG
  success = GPU_pass_compile(mat->pass, mat->name);
#else
  success = GPU_pass_compile(mat->pass, __func__);
#endif

  mat->flag |= GPU_MATFLAG_UPDATED;

  if (success) {
    GPUShader *sh = GPU_pass_shader_get(mat->pass);
    if (sh != NULL) {
      mat->status = GPU_MAT_SUCCESS;
      gpu_node_graph_free_nodes(&mat->graph);
    }
  }
  else {
    mat->status = GPU_MAT_FAILED;
    GPU_pass_release(mat->pass);
    mat->pass = NULL;
    gpu_node_graph_free(&mat->graph);
  }
}

void GPU_materials_free(Main *bmain)
{
  LISTBASE_FOREACH (Material *, ma, &bmain->materials) {
    GPU_material_free(&ma->gpumaterial);
  }

  LISTBASE_FOREACH (World *, wo, &bmain->worlds) {
    GPU_material_free(&wo->gpumaterial);
  }

  BKE_world_defaults_free_gpu();
  BKE_material_defaults_free_gpu();
}
