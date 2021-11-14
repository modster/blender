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

#include "../node_composite_util.hh"

/* **************** SCALAR MATH ******************** */
static bNodeSocketTemplate cmp_node_ellipsemask_in[] = {
    {SOCK_FLOAT, N_("Mask"), 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f},
    {SOCK_FLOAT, N_("Value"), 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f},
    {-1, ""}};

static bNodeSocketTemplate cmp_node_ellipsemask_out[] = {
    {SOCK_FLOAT, N_("Mask"), 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f}, {-1, ""}};

static void node_composit_init_ellipsemask(bNodeTree *UNUSED(ntree), bNode *node)
{
  NodeEllipseMask *data = (NodeEllipseMask *)MEM_callocN(sizeof(NodeEllipseMask),
                                                         "NodeEllipseMask");
  data->x = 0.5;
  data->y = 0.5;
  data->width = 0.2;
  data->height = 0.1;
  data->rotation = 0.0;
  node->storage = data;
}

static int node_composite_gpu_ellipsemask(GPUMaterial *mat,
                                          bNode *node,
                                          bNodeExecData *UNUSED(execdata),
                                          GPUNodeStack *in,
                                          GPUNodeStack *out)
{
  const NodeEllipseMask *data = (NodeEllipseMask *)node->storage;

  const float mask_type = (float)node->custom1;
  const float cos_angle = std::cos(data->rotation);
  const float sin_angle = std::sin(data->rotation);
  const float half_width = data->width / 2.0;
  const float half_height = data->height / 2.0;

  return GPU_stack_link(mat,
                        node,
                        "node_composite_ellipse_mask",
                        in,
                        out,
                        GPU_constant(&mask_type),
                        GPU_uniform(&data->x),
                        GPU_uniform(&data->y),
                        GPU_uniform(&half_width),
                        GPU_uniform(&half_height),
                        GPU_uniform(&cos_angle),
                        GPU_uniform(&sin_angle));
}

void register_node_type_cmp_ellipsemask(void)
{
  static bNodeType ntype;

  cmp_node_type_base(&ntype, CMP_NODE_MASK_ELLIPSE, "Ellipse Mask", NODE_CLASS_MATTE, 0);
  node_type_socket_templates(&ntype, cmp_node_ellipsemask_in, cmp_node_ellipsemask_out);
  node_type_size(&ntype, 260, 110, 320);
  node_type_init(&ntype, node_composit_init_ellipsemask);
  node_type_storage(
      &ntype, "NodeEllipseMask", node_free_standard_storage, node_copy_standard_storage);
  node_type_gpu(&ntype, node_composite_gpu_ellipsemask);

  nodeRegisterType(&ntype);
}
