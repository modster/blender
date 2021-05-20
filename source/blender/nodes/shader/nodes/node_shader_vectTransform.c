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
 * The Original Code is Copyright (C) 2013 Blender Foundation.
 * All rights reserved.
 */

/** \file
 * \ingroup shdnodes
 */

#include "BLI_string.h"

#include "../node_shader_util.h"

/* **************** Vector Transform ******************** */
static bNodeSocketTemplate sh_node_vect_transform_in[] = {
    {SOCK_VECTOR, N_("Vector"), 0.5f, 0.5f, 0.5f, 1.0f, -10000.0f, 10000.0f, PROP_NONE}, {-1, ""}};

static bNodeSocketTemplate sh_node_vect_transform_out[] = {
    {SOCK_VECTOR, N_("Vector")},
    {-1, ""},
};

static void node_shader_init_vect_transform(bNodeTree *UNUSED(ntree), bNode *node)
{
  NodeShaderVectTransform *vect = MEM_callocN(sizeof(NodeShaderVectTransform),
                                              "NodeShaderVectTransform");

  /* Convert World into Object Space per default */
  vect->convert_to = 1;

  node->storage = vect;
}

static void node_shader_exec_vect_transform(void *UNUSED(data),
                                            int UNUSED(thread),
                                            bNode *UNUSED(node),
                                            bNodeExecData *UNUSED(execdata),
                                            bNodeStack **UNUSED(in),
                                            bNodeStack **UNUSED(out))
{
}

static const char *get_gpufn_name_from_to(short from, short to)
{
  switch (from) {
    case SHD_VECT_TRANSFORM_SPACE_OBJECT:
      switch (to) {
        case SHD_VECT_TRANSFORM_SPACE_OBJECT:
          return NULL;
        case SHD_VECT_TRANSFORM_SPACE_WORLD:
          return "object_to_world";
        case SHD_VECT_TRANSFORM_SPACE_CAMERA:
          return "object_to_view";
      }
      break;
    case SHD_VECT_TRANSFORM_SPACE_WORLD:
      switch (to) {
        case SHD_VECT_TRANSFORM_SPACE_WORLD:
          return NULL;
        case SHD_VECT_TRANSFORM_SPACE_CAMERA:
          return "world_to_view";
        case SHD_VECT_TRANSFORM_SPACE_OBJECT:
          return "world_to_object";
      }
      break;
    case SHD_VECT_TRANSFORM_SPACE_CAMERA:
      switch (to) {
        case SHD_VECT_TRANSFORM_SPACE_CAMERA:
          return NULL;
        case SHD_VECT_TRANSFORM_SPACE_WORLD:
          return "view_to_world";
        case SHD_VECT_TRANSFORM_SPACE_OBJECT:
          return "view_to_object";
      }
      break;
  }
  return NULL;
}

static int gpu_shader_vect_transform(GPUMaterial *mat,
                                     bNode *node,
                                     bNodeExecData *UNUSED(execdata),
                                     GPUNodeStack *in,
                                     GPUNodeStack *out)
{
  struct GPUNodeLink *inputlink;

  NodeShaderVectTransform *nodeprop = (NodeShaderVectTransform *)node->storage;

  if (in[0].hasinput) {
    inputlink = in[0].link;
  }
  else {
    inputlink = GPU_constant(in[0].vec);
  }

  const char *xform = (nodeprop->type == SHD_VECT_TRANSFORM_TYPE_POINT) ? "point_transform_" :
                                                                          "direction_transform_";
  const char *fromto = get_gpufn_name_from_to(nodeprop->convert_from, nodeprop->convert_to);

  if (fromto) {
    /* For cycles we have inverted Z */
    /* TODO: pass here the correct matrices */
    if (nodeprop->convert_from == SHD_VECT_TRANSFORM_SPACE_CAMERA &&
        nodeprop->convert_to != SHD_VECT_TRANSFORM_SPACE_CAMERA) {
      GPU_link(mat, "invert_z", inputlink, &inputlink);
    }

    char func_name[48];
    SNPRINTF(func_name, "%s%s", xform, fromto);
    GPU_link(mat, func_name, inputlink, &out[0].link);

    if (nodeprop->convert_to == SHD_VECT_TRANSFORM_SPACE_CAMERA &&
        nodeprop->convert_from != SHD_VECT_TRANSFORM_SPACE_CAMERA) {
      GPU_link(mat, "invert_z", out[0].link, &out[0].link);
    }
  }
  else {
    GPU_link(mat, "set_rgb", inputlink, &out[0].link);
  }

  if (nodeprop->type == SHD_VECT_TRANSFORM_TYPE_NORMAL) {
    GPU_link(mat, "vector_normalize", out[0].link, &out[0].link);
  }

  return true;
}

void register_node_type_sh_vect_transform(void)
{
  static bNodeType ntype;

  sh_node_type_base(&ntype, SH_NODE_VECT_TRANSFORM, "Vector Transform", NODE_CLASS_OP_VECTOR, 0);
  node_type_init(&ntype, node_shader_init_vect_transform);
  node_type_socket_templates(&ntype, sh_node_vect_transform_in, sh_node_vect_transform_out);
  node_type_storage(
      &ntype, "NodeShaderVectTransform", node_free_standard_storage, node_copy_standard_storage);
  node_type_exec(&ntype, NULL, NULL, node_shader_exec_vect_transform);
  node_type_gpu(&ntype, gpu_shader_vect_transform);

  nodeRegisterType(&ntype);
}
