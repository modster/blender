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

/* ******************* Channel Matte Node ********************************* */
static bNodeSocketTemplate cmp_node_channel_matte_in[] = {
    {SOCK_RGBA, N_("Image"), 1.0f, 1.0f, 1.0f, 1.0f},
    {-1, ""},
};

static bNodeSocketTemplate cmp_node_channel_matte_out[] = {
    {SOCK_RGBA, N_("Image")},
    {SOCK_FLOAT, N_("Matte")},
    {-1, ""},
};

static void node_composit_init_channel_matte(bNodeTree *UNUSED(ntree), bNode *node)
{
  NodeChroma *c = (NodeChroma *)MEM_callocN(sizeof(NodeChroma), "node chroma");
  node->storage = c;
  c->t1 = 1.0f;
  c->t2 = 0.0f;
  c->t3 = 0.0f;
  c->fsize = 0.0f;
  c->fstrength = 0.0f;
  c->algorithm = 1;  /* Max channel limiting. */
  c->channel = 1;    /* Limit by red. */
  node->custom1 = 1; /* RGB channel. */
  node->custom2 = 2; /* Green Channel. */
}

static int node_composite_gpu_channel_matte(GPUMaterial *mat,
                                            bNode *node,
                                            bNodeExecData *UNUSED(execdata),
                                            GPUNodeStack *in,
                                            GPUNodeStack *out)
{
  const NodeChroma *data = (NodeChroma *)node->storage;

  const float color_space = (float)node->custom1;
  const float matte_channel = (float)(node->custom2 - 1);

  /* Always assume the limit algorithm is Max, if it is a single limit channel, store it in both
   * limit channels. */
  float limit_channels[2];
  if (data->algorithm == 1) {
    limit_channels[0] = (float)(node->custom2 % 3);
    limit_channels[1] = (float)((node->custom2 + 1) % 3);
  }
  else {
    limit_channels[0] = (float)(data->channel - 1);
    limit_channels[1] = (float)(data->channel - 1);
  }

  return GPU_stack_link(mat,
                        node,
                        "node_composite_channel_matte",
                        in,
                        out,
                        GPU_constant(&color_space),
                        GPU_constant(&matte_channel),
                        GPU_constant(limit_channels),
                        GPU_uniform(&data->t1),
                        GPU_uniform(&data->t2));
}

void register_node_type_cmp_channel_matte(void)
{
  static bNodeType ntype;

  cmp_node_type_base(
      &ntype, CMP_NODE_CHANNEL_MATTE, "Channel Key", NODE_CLASS_MATTE, NODE_PREVIEW);
  node_type_socket_templates(&ntype, cmp_node_channel_matte_in, cmp_node_channel_matte_out);
  node_type_init(&ntype, node_composit_init_channel_matte);
  node_type_storage(&ntype, "NodeChroma", node_free_standard_storage, node_copy_standard_storage);
  node_type_gpu(&ntype, node_composite_gpu_channel_matte);

  nodeRegisterType(&ntype);
}
