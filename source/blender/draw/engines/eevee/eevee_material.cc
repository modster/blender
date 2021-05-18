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
 */

#include "DNA_material_types.h"

#include "BKE_lib_id.h"
#include "BKE_material.h"
#include "BKE_node.h"
#include "NOD_shader.h"

#include "eevee_instance.hh"

#include "eevee_material.hh"

namespace blender::eevee {

/* -------------------------------------------------------------------- */
/** \name Default Material
 *
 * \{ */

DefaultSurfaceNodeTree::DefaultSurfaceNodeTree()
{
  bNodeTree *ntree = ntreeAddTree(NULL, "Shader Nodetree", ntreeType_Shader->idname);
  bNode *bsdf = nodeAddStaticNode(NULL, ntree, SH_NODE_BSDF_PRINCIPLED);
  bNode *output = nodeAddStaticNode(NULL, ntree, SH_NODE_OUTPUT_MATERIAL);
  bNodeSocket *bsdf_out = nodeFindSocket(bsdf, SOCK_OUT, "BSDF");
  bNodeSocket *output_in = nodeFindSocket(output, SOCK_IN, "Surface");
  nodeAddLink(ntree, bsdf, bsdf_out, output, output_in);
  nodeSetActive(ntree, output);

  color_socket_ =
      (bNodeSocketValueRGBA *)nodeFindSocket(bsdf, SOCK_IN, "Base Color")->default_value;
  metallic_socket_ =
      (bNodeSocketValueFloat *)nodeFindSocket(bsdf, SOCK_IN, "Metallic")->default_value;
  roughness_socket_ =
      (bNodeSocketValueFloat *)nodeFindSocket(bsdf, SOCK_IN, "Roughness")->default_value;
  specular_socket_ =
      (bNodeSocketValueFloat *)nodeFindSocket(bsdf, SOCK_IN, "Specular")->default_value;
  ntree_ = ntree;
}

DefaultSurfaceNodeTree::~DefaultSurfaceNodeTree()
{
  ntreeFreeEmbeddedTree(ntree_);
  MEM_SAFE_FREE(ntree_);
}

/* Configure a default nodetree with the given material.  */
bNodeTree *DefaultSurfaceNodeTree::nodetree_get(::Material *ma)
{
  /* WARNING: This function is not threadsafe. Which is not a problem for the moment. */
  copy_v3_fl3(color_socket_->value, ma->r, ma->g, ma->b);
  metallic_socket_->value = ma->metallic;
  roughness_socket_->value = ma->roughness;
  specular_socket_->value = ma->spec;

  return ntree_;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Material
 *
 * \{ */

MaterialModule::MaterialModule(Instance &inst) : inst_(inst)
{
  {
    bNodeTree *ntree = ntreeAddTree(NULL, "Shader Nodetree", ntreeType_Shader->idname);

    diffuse_mat_ = (::Material *)BKE_id_new_nomain(ID_MA, "EEVEE default diffuse");
    diffuse_mat_->nodetree = ntree;
    diffuse_mat_->use_nodes = true;

    bNode *bsdf = nodeAddStaticNode(NULL, ntree, SH_NODE_BSDF_DIFFUSE);
    bNodeSocket *base_color = nodeFindSocket(bsdf, SOCK_IN, "Color");
    copy_v3_fl(((bNodeSocketValueRGBA *)base_color->default_value)->value, 0.8f);

    bNode *output = nodeAddStaticNode(NULL, ntree, SH_NODE_OUTPUT_MATERIAL);

    nodeAddLink(ntree,
                bsdf,
                nodeFindSocket(bsdf, SOCK_OUT, "BSDF"),
                output,
                nodeFindSocket(output, SOCK_IN, "Surface"));

    nodeSetActive(ntree, output);
  }
  {
    bNodeTree *ntree = ntreeAddTree(NULL, "Shader Nodetree", ntreeType_Shader->idname);

    glossy_mat_ = (::Material *)BKE_id_new_nomain(ID_MA, "EEVEE default metal");
    glossy_mat_->nodetree = ntree;
    glossy_mat_->use_nodes = true;

    bNode *bsdf = nodeAddStaticNode(NULL, ntree, SH_NODE_BSDF_GLOSSY);
    bNodeSocket *base_color = nodeFindSocket(bsdf, SOCK_IN, "Color");
    copy_v3_fl(((bNodeSocketValueRGBA *)base_color->default_value)->value, 1.0f);
    bNodeSocket *roughness = nodeFindSocket(bsdf, SOCK_IN, "Roughness");
    ((bNodeSocketValueFloat *)roughness->default_value)->value = 0.0f;

    bNode *output = nodeAddStaticNode(NULL, ntree, SH_NODE_OUTPUT_MATERIAL);

    nodeAddLink(ntree,
                bsdf,
                nodeFindSocket(bsdf, SOCK_OUT, "BSDF"),
                output,
                nodeFindSocket(output, SOCK_IN, "Surface"));

    nodeSetActive(ntree, output);
  }
  {
    bNodeTree *ntree = ntreeAddTree(NULL, "Shader Nodetree", ntreeType_Shader->idname);

    error_mat_ = (::Material *)BKE_id_new_nomain(ID_MA, "EEVEE default error");
    error_mat_->nodetree = ntree;
    error_mat_->use_nodes = true;

    /* Use emission and output material to be compatible with both World and Material. */
    bNode *bsdf = nodeAddStaticNode(NULL, ntree, SH_NODE_EMISSION);
    bNodeSocket *color = nodeFindSocket(bsdf, SOCK_IN, "Color");
    copy_v3_fl3(((bNodeSocketValueRGBA *)color->default_value)->value, 1.0f, 0.0f, 1.0f);

    bNode *output = nodeAddStaticNode(NULL, ntree, SH_NODE_OUTPUT_MATERIAL);

    nodeAddLink(ntree,
                bsdf,
                nodeFindSocket(bsdf, SOCK_OUT, "Emission"),
                output,
                nodeFindSocket(output, SOCK_IN, "Surface"));

    nodeSetActive(ntree, output);
  }
}

MaterialModule::~MaterialModule()
{
  BKE_id_free(NULL, glossy_mat_);
  BKE_id_free(NULL, diffuse_mat_);
  BKE_id_free(NULL, error_mat_);
}

void MaterialModule::begin_sync(void)
{
  queued_shaders_count_ = 0;

  for (Material &mat : material_map_.values()) {
    mat.shading.shgrp = nullptr;
    mat.shadow.shgrp = nullptr;
  }
  for (DRWShadingGroup *&shgroup : shader_map_.values()) {
    shgroup = nullptr;
  }
}

Material &MaterialModule::material_sync(::Material *blender_mat, eMaterialGeometry geometry_type)
{
  MaterialKey material_key(blender_mat, geometry_type);

  Material &mat = material_map_.lookup_or_add_default(material_key);

  if (mat.shading.shgrp == nullptr) {
    mat.shading = material_pass_get(blender_mat, geometry_type, MAT_DOMAIN_SURFACE);
    mat.shadow = material_pass_get(blender_mat, geometry_type, MAT_DOMAIN_SHADOW);
  }
  return mat;
}

MaterialArray &MaterialModule::surface_materials_get(Object *ob)
{
  material_array_.materials.clear();
  material_array_.gpu_materials.clear();

  for (auto i : IndexRange(ob->totcol)) {
    ::Material *blender_mat = material_from_slot(ob, i);
    Material &mat = material_sync(blender_mat, to_material_geometry(ob));
    material_array_.materials.append(&mat);
    material_array_.gpu_materials.append(mat.shading.gpumat);
  }
  return material_array_;
}

/* Return correct material or empty default material if slot is empty. */
::Material *MaterialModule::material_from_slot(Object *ob, int slot)
{
  if (ob->base_flag & BASE_HOLDOUT) {
    return BKE_material_default_holdout();
  }
  ::Material *ma = BKE_object_material_get(ob, slot + 1);
  if (ma == nullptr || !ma->use_nodes || ma->nodetree == nullptr) {
    if (ob->type == OB_VOLUME) {
      ma = BKE_material_default_volume();
    }
    else {
      ma = BKE_material_default_surface();
    }
  }
  return ma;
}

MaterialPass MaterialModule::material_pass_get(::Material *blender_mat,
                                               eMaterialGeometry geometry_type,
                                               eMaterialDomain domain_type)
{
  MaterialPass matpass;
  matpass.gpumat = inst_.shaders.material_shader_get(
      inst_.scene, blender_mat, geometry_type, domain_type, true);

  ShaderKey shader_key(matpass.gpumat, blender_mat, geometry_type, domain_type);

  switch (GPU_material_status(matpass.gpumat)) {
    case GPU_MAT_SUCCESS:
      break;
    case GPU_MAT_QUEUED:
      queued_shaders_count_++;
      matpass.gpumat = inst_.shaders.material_shader_get(inst_.scene,
                                                         (geometry_type == MAT_GEOM_VOLUME) ?
                                                             BKE_material_default_volume() :
                                                             BKE_material_default_surface(),
                                                         geometry_type,
                                                         domain_type,
                                                         false);
      break;
    case GPU_MAT_FAILED:
    default:
      matpass.gpumat = inst_.shaders.material_shader_get(
          inst_.scene, error_mat_, geometry_type, domain_type, false);
      break;
  }
  /* Returned material should be ready to be drawn. */
  BLI_assert(GPU_material_status(matpass.gpumat) == GPU_MAT_SUCCESS);

  matpass.shgrp = &shader_map_.lookup_or_add_default(shader_key);

  if (*matpass.shgrp != nullptr) {
    /* Shading group for this shader already exists. Create a sub one for this material. */
    *matpass.shgrp = DRW_shgroup_create_sub(*matpass.shgrp);
    DRW_shgroup_add_material_resources(*matpass.shgrp, matpass.gpumat);
  }
  return matpass;
}

/** \} */

}  // namespace blender::eevee