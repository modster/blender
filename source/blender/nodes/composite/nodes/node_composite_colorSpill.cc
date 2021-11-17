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

/* ******************* Color Spill Suppression ********************************* */
static bNodeSocketTemplate cmp_node_color_spill_in[] = {
    {SOCK_RGBA, N_("Image"), 1.0f, 1.0f, 1.0f, 1.0f},
    {SOCK_FLOAT, N_("Fac"), 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, PROP_FACTOR},
    {-1, ""},
};

static bNodeSocketTemplate cmp_node_color_spill_out[] = {
    {SOCK_RGBA, N_("Image")},
    {-1, ""},
};

static void node_composit_init_color_spill(bNodeTree *UNUSED(ntree), bNode *node)
{
  NodeColorspill *ncs = (NodeColorspill *)MEM_callocN(sizeof(NodeColorspill), "node colorspill");
  node->storage = ncs;
  node->custom1 = 2;    /* green channel */
  node->custom2 = 0;    /* simple limit algorithm */
  ncs->limchan = 0;     /* limit by red */
  ncs->limscale = 1.0f; /* limit scaling factor */
  ncs->unspill = 0;     /* do not use unspill */
}

static int node_composite_gpu_color_spill(GPUMaterial *mat,
                                          bNode *node,
                                          bNodeExecData *UNUSED(execdata),
                                          GPUNodeStack *in,
                                          GPUNodeStack *out)
{
  const NodeColorspill *data = (NodeColorspill *)node->storage;

  const float spill_channel = (float)(node->custom1 - 1);
  float spill_scale[3] = {data->uspillr, data->uspillg, data->uspillb};
  spill_scale[node->custom1 - 1] *= -1.0f;
  if (data->unspill == 0) {
    spill_scale[0] = 0.0f;
    spill_scale[1] = 0.0f;
    spill_scale[2] = 0.0f;
    spill_scale[node->custom1 - 1] = -1.0f;
  }

  /* Always assume the limit method to be average, and for the single method, assign the same
   * channel to both limit channels. */
  float limit_channels[2];
  if (node->custom2 == 0) {
    limit_channels[0] = (float)data->limchan;
    limit_channels[1] = (float)data->limchan;
  }
  else {
    limit_channels[0] = (float)(node->custom1 % 3);
    limit_channels[1] = (float)((node->custom1 + 1) % 3);
  }
  const float limit_scale = data->limscale;

  return GPU_stack_link(mat,
                        node,
                        "node_composite_color_spill",
                        in,
                        out,
                        GPU_uniform(&spill_channel),
                        GPU_uniform(spill_scale),
                        GPU_uniform(limit_channels),
                        GPU_uniform(&limit_scale));
}

void register_node_type_cmp_color_spill(void)
{
  static bNodeType ntype;

  cmp_node_type_base(&ntype, CMP_NODE_COLOR_SPILL, "Color Spill", NODE_CLASS_MATTE, 0);
  node_type_socket_templates(&ntype, cmp_node_color_spill_in, cmp_node_color_spill_out);
  node_type_init(&ntype, node_composit_init_color_spill);
  node_type_storage(
      &ntype, "NodeColorspill", node_free_standard_storage, node_copy_standard_storage);
  node_type_gpu(&ntype, node_composite_gpu_color_spill);

  nodeRegisterType(&ntype);
}
