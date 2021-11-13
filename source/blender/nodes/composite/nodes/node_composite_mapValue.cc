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
 * \ingroup cmpnodes
 */

#include "node_composite_util.hh"

/* **************** MAP VALUE ******************** */
static bNodeSocketTemplate cmp_node_map_value_in[] = {
    {SOCK_FLOAT, N_("Value"), 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, PROP_NONE},
    {-1, ""},
};
static bNodeSocketTemplate cmp_node_map_value_out[] = {
    {SOCK_FLOAT, N_("Value")},
    {-1, ""},
};

static void node_composit_init_map_value(bNodeTree *UNUSED(ntree), bNode *node)
{
  node->storage = BKE_texture_mapping_add(TEXMAP_TYPE_POINT);
}

static int node_composite_gpu_map_value(GPUMaterial *mat,
                                        bNode *node,
                                        bNodeExecData *UNUSED(execdata),
                                        GPUNodeStack *in,
                                        GPUNodeStack *out)
{
  const TexMapping *texture_mapping = (TexMapping *)node->storage;

  const float use_min = texture_mapping->flag & TEXMAP_CLIP_MIN ? 1.0f : 0.0f;
  const float use_max = texture_mapping->flag & TEXMAP_CLIP_MAX ? 1.0f : 0.0f;

  return GPU_stack_link(mat,
                        node,
                        "node_composite_map_value",
                        in,
                        out,
                        GPU_uniform(texture_mapping->loc),
                        GPU_uniform(texture_mapping->size),
                        GPU_constant(&use_min),
                        GPU_uniform(texture_mapping->min),
                        GPU_constant(&use_max),
                        GPU_uniform(texture_mapping->max));
}

void register_node_type_cmp_map_value(void)
{
  static bNodeType ntype;

  cmp_node_type_base(&ntype, CMP_NODE_MAP_VALUE, "Map Value", NODE_CLASS_OP_VECTOR, 0);
  node_type_socket_templates(&ntype, cmp_node_map_value_in, cmp_node_map_value_out);
  node_type_init(&ntype, node_composit_init_map_value);
  node_type_storage(&ntype, "TexMapping", node_free_standard_storage, node_copy_standard_storage);
  node_type_gpu(&ntype, node_composite_gpu_map_value);

  nodeRegisterType(&ntype);
}
