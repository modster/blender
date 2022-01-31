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

#include "RNA_access.h"

#include "UI_interface.h"
#include "UI_resources.h"

#include "node_composite_util.hh"

/* ******************* Color Spill Suppression ********************************* */

namespace blender::nodes::node_composite_color_spill_cc {

static void cmp_node_color_spill_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Color>(N_("Image")).default_value({1.0f, 1.0f, 1.0f, 1.0f});
  b.add_input<decl::Float>(N_("Fac")).default_value(1.0f).min(0.0f).max(1.0f).subtype(PROP_FACTOR);
  b.add_output<decl::Color>(N_("Image"));
}

static void node_composit_init_color_spill(bNodeTree *UNUSED(ntree), bNode *node)
{
  NodeColorspill *ncs = MEM_cnew<NodeColorspill>(__func__);
  node->storage = ncs;
  node->custom1 = 2;    /* green channel */
  node->custom2 = 0;    /* simple limit algorithm */
  ncs->limchan = 0;     /* limit by red */
  ncs->limscale = 1.0f; /* limit scaling factor */
  ncs->unspill = 0;     /* do not use unspill */
}

static void node_composit_buts_color_spill(uiLayout *layout, bContext *UNUSED(C), PointerRNA *ptr)
{
  uiLayout *row, *col;

  uiItemL(layout, IFACE_("Despill Channel:"), ICON_NONE);
  row = uiLayoutRow(layout, false);
  uiItemR(row, ptr, "channel", UI_ITEM_R_SPLIT_EMPTY_NAME | UI_ITEM_R_EXPAND, nullptr, ICON_NONE);

  col = uiLayoutColumn(layout, false);
  uiItemR(col, ptr, "limit_method", UI_ITEM_R_SPLIT_EMPTY_NAME, nullptr, ICON_NONE);

  if (RNA_enum_get(ptr, "limit_method") == 0) {
    uiItemL(col, IFACE_("Limiting Channel:"), ICON_NONE);
    row = uiLayoutRow(col, false);
    uiItemR(row,
            ptr,
            "limit_channel",
            UI_ITEM_R_SPLIT_EMPTY_NAME | UI_ITEM_R_EXPAND,
            nullptr,
            ICON_NONE);
  }

  uiItemR(col, ptr, "ratio", UI_ITEM_R_SPLIT_EMPTY_NAME | UI_ITEM_R_SLIDER, nullptr, ICON_NONE);
  uiItemR(col, ptr, "use_unspill", UI_ITEM_R_SPLIT_EMPTY_NAME, nullptr, ICON_NONE);
  if (RNA_boolean_get(ptr, "use_unspill") == true) {
    uiItemR(col,
            ptr,
            "unspill_red",
            UI_ITEM_R_SPLIT_EMPTY_NAME | UI_ITEM_R_SLIDER,
            nullptr,
            ICON_NONE);
    uiItemR(col,
            ptr,
            "unspill_green",
            UI_ITEM_R_SPLIT_EMPTY_NAME | UI_ITEM_R_SLIDER,
            nullptr,
            ICON_NONE);
    uiItemR(col,
            ptr,
            "unspill_blue",
            UI_ITEM_R_SPLIT_EMPTY_NAME | UI_ITEM_R_SLIDER,
            nullptr,
            ICON_NONE);
  }
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

}  // namespace blender::nodes::node_composite_color_spill_cc

void register_node_type_cmp_color_spill()
{
  namespace file_ns = blender::nodes::node_composite_color_spill_cc;

  static bNodeType ntype;

  cmp_node_type_base(&ntype, CMP_NODE_COLOR_SPILL, "Color Spill", NODE_CLASS_MATTE);
  ntype.declare = file_ns::cmp_node_color_spill_declare;
  ntype.draw_buttons = file_ns::node_composit_buts_color_spill;
  node_type_init(&ntype, file_ns::node_composit_init_color_spill);
  node_type_storage(
      &ntype, "NodeColorspill", node_free_standard_storage, node_copy_standard_storage);
  node_type_gpu(&ntype, file_ns::node_composite_gpu_color_spill);

  nodeRegisterType(&ntype);
}
