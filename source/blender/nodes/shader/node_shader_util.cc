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
 * \ingroup nodes
 */

#include "DNA_node_types.h"

#include "node_shader_util.hh"

#include "NOD_socket_search_link.hh"

#include "node_exec.h"

bool sh_node_poll_default(bNodeType *UNUSED(ntype), bNodeTree *ntree, const char **r_disabled_hint)
{
  if (!STREQ(ntree->idname, "ShaderNodeTree")) {
    *r_disabled_hint = TIP_("Not a shader node tree");
    return false;
  }
  return true;
}

static bool sh_fn_poll_default(bNodeType *UNUSED(ntype),
                               bNodeTree *ntree,
                               const char **r_disabled_hint)
{
  if (!STR_ELEM(ntree->idname, "ShaderNodeTree", "GeometryNodeTree")) {
    *r_disabled_hint = TIP_("Not a shader or geometry node tree");
    return false;
  }
  return true;
}

void sh_node_type_base(struct bNodeType *ntype, int type, const char *name, short nclass)
{
  node_type_base(ntype, type, name, nclass);

  ntype->poll = sh_node_poll_default;
  ntype->insert_link = node_insert_link_default;
  ntype->gather_link_search_ops = blender::nodes::search_link_ops_for_basic_node;
}

void sh_fn_node_type_base(bNodeType *ntype, int type, const char *name, short nclass)
{
  sh_node_type_base(ntype, type, name, nclass);
  ntype->poll = sh_fn_poll_default;
  ntype->gather_link_search_ops = blender::nodes::search_link_ops_for_basic_node;
}

bNode *nodeGetActiveTexture(bNodeTree *ntree)
{
  /* this is the node we texture paint and draw in textured draw */
  bNode *inactivenode = nullptr, *activetexnode = nullptr, *activegroup = nullptr;
  bool hasgroup = false;

  if (!ntree) {
    return nullptr;
  }

  LISTBASE_FOREACH (bNode *, node, &ntree->nodes) {
    if (node->flag & NODE_ACTIVE_TEXTURE) {
      activetexnode = node;
      /* if active we can return immediately */
      if (node->flag & NODE_ACTIVE) {
        return node;
      }
    }
    else if (!inactivenode && node->typeinfo->nclass == NODE_CLASS_TEXTURE) {
      inactivenode = node;
    }
    else if (node->type == NODE_GROUP) {
      if (node->flag & NODE_ACTIVE) {
        activegroup = node;
      }
      else {
        hasgroup = true;
      }
    }
  }

  /* first, check active group for textures */
  if (activegroup) {
    bNode *tnode = nodeGetActiveTexture((bNodeTree *)activegroup->id);
    /* active node takes priority, so ignore any other possible nodes here */
    if (tnode) {
      return tnode;
    }
  }

  if (activetexnode) {
    return activetexnode;
  }

  if (hasgroup) {
    /* node active texture node in this tree, look inside groups */
    LISTBASE_FOREACH (bNode *, node, &ntree->nodes) {
      if (node->type == NODE_GROUP) {
        bNode *tnode = nodeGetActiveTexture((bNodeTree *)node->id);
        if (tnode && ((tnode->flag & NODE_ACTIVE_TEXTURE) || !inactivenode)) {
          return tnode;
        }
      }
    }
  }

  return inactivenode;
}

void node_shader_gpu_bump_tex_coord(GPUMaterial *mat, bNode *node, GPUNodeLink **link)
{
  if (node->branch_tag == 1) {
    /* Add one time the value for derivative to the input vector. */
    GPU_link(mat, "dfdx_v3", *link, link);
  }
  else if (node->branch_tag == 2) {
    /* Add one time the value for derivative to the input vector. */
    GPU_link(mat, "dfdy_v3", *link, link);
  }
  else {
    /* nothing to do, reference center value. */
  }
}

void node_shader_gpu_default_tex_coord(GPUMaterial *mat, bNode *node, GPUNodeLink **link)
{
  if (!*link) {
    *link = GPU_attribute(mat, CD_ORCO, "");
    GPU_link(mat, "generated_texco", *link, link);
    node_shader_gpu_bump_tex_coord(mat, node, link);
  }
}

void node_shader_gpu_tex_mapping(GPUMaterial *mat,
                                 bNode *node,
                                 GPUNodeStack *in,
                                 GPUNodeStack *UNUSED(out))
{
  NodeTexBase *base = (NodeTexBase *)node->storage;
  TexMapping *texmap = &base->tex_mapping;
  float domin = (texmap->flag & TEXMAP_CLIP_MIN) != 0;
  float domax = (texmap->flag & TEXMAP_CLIP_MAX) != 0;

  if (domin || domax || !(texmap->flag & TEXMAP_UNIT_MATRIX)) {
    static float max[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
    static float min[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    GPUNodeLink *tmin, *tmax, *tmat0, *tmat1, *tmat2, *tmat3;

    tmin = GPU_uniform((domin) ? texmap->min : min);
    tmax = GPU_uniform((domax) ? texmap->max : max);
    tmat0 = GPU_uniform((float *)texmap->mat[0]);
    tmat1 = GPU_uniform((float *)texmap->mat[1]);
    tmat2 = GPU_uniform((float *)texmap->mat[2]);
    tmat3 = GPU_uniform((float *)texmap->mat[3]);

    GPU_link(mat, "mapping_mat4", in[0].link, tmat0, tmat1, tmat2, tmat3, tmin, tmax, &in[0].link);

    if (texmap->type == TEXMAP_TYPE_NORMAL) {
      GPU_link(mat, "vector_normalize", in[0].link, &in[0].link);
    }
  }
}

void get_XYZ_to_RGB_for_gpu(XYZ_to_RGB *data)
{
  const float *xyz_to_rgb = IMB_colormanagement_get_xyz_to_rgb();
  data->r[0] = xyz_to_rgb[0];
  data->r[1] = xyz_to_rgb[3];
  data->r[2] = xyz_to_rgb[6];
  data->g[0] = xyz_to_rgb[1];
  data->g[1] = xyz_to_rgb[4];
  data->g[2] = xyz_to_rgb[7];
  data->b[0] = xyz_to_rgb[2];
  data->b[1] = xyz_to_rgb[5];
  data->b[2] = xyz_to_rgb[8];
}
