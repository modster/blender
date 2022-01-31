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

#include "UI_interface.h"
#include "UI_resources.h"

#include "node_composite_util.hh"

/* ******************* Chroma Key ********************************************************** */

namespace blender::nodes::node_composite_chroma_matte_cc {

static void cmp_node_chroma_matte_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Color>(N_("Image")).default_value({1.0f, 1.0f, 1.0f, 1.0f});
  b.add_input<decl::Color>(N_("Key Color")).default_value({1.0f, 1.0f, 1.0f, 1.0f});
  b.add_output<decl::Color>(N_("Image"));
  b.add_output<decl::Float>(N_("Matte"));
}

static void node_composit_init_chroma_matte(bNodeTree *UNUSED(ntree), bNode *node)
{
  NodeChroma *c = MEM_cnew<NodeChroma>(__func__);
  node->storage = c;
  c->t1 = DEG2RADF(30.0f);
  c->t2 = DEG2RADF(10.0f);
  c->t3 = 0.0f;
  c->fsize = 0.0f;
  c->fstrength = 1.0f;
}

static void node_composit_buts_chroma_matte(uiLayout *layout, bContext *UNUSED(C), PointerRNA *ptr)
{
  uiLayout *col;

  col = uiLayoutColumn(layout, false);
  uiItemR(col, ptr, "tolerance", UI_ITEM_R_SPLIT_EMPTY_NAME, nullptr, ICON_NONE);
  uiItemR(col, ptr, "threshold", UI_ITEM_R_SPLIT_EMPTY_NAME, nullptr, ICON_NONE);

  col = uiLayoutColumn(layout, true);
  /* Removed for now. */
  // uiItemR(col, ptr, "lift", UI_ITEM_R_SLIDER, nullptr, ICON_NONE);
  uiItemR(col, ptr, "gain", UI_ITEM_R_SPLIT_EMPTY_NAME | UI_ITEM_R_SLIDER, nullptr, ICON_NONE);
  /* Removed for now. */
  // uiItemR(col, ptr, "shadow_adjust", UI_ITEM_R_SLIDER, nullptr, ICON_NONE);
}

static int node_composite_gpu_chroma_matte(GPUMaterial *mat,
                                           bNode *node,
                                           bNodeExecData *UNUSED(execdata),
                                           GPUNodeStack *in,
                                           GPUNodeStack *out)
{
  const NodeChroma *data = (NodeChroma *)node->storage;

  const float acceptance = std::tan(data->t1) / 2.0f;

  return GPU_stack_link(mat,
                        node,
                        "node_composite_chroma_matte",
                        in,
                        out,
                        GPU_uniform(&acceptance),
                        GPU_uniform(&data->t2),
                        GPU_uniform(&data->fstrength));
}

}  // namespace blender::nodes::node_composite_chroma_matte_cc

void register_node_type_cmp_chroma_matte()
{
  namespace file_ns = blender::nodes::node_composite_chroma_matte_cc;

  static bNodeType ntype;

  cmp_node_type_base(&ntype, CMP_NODE_CHROMA_MATTE, "Chroma Key", NODE_CLASS_MATTE);
  ntype.declare = file_ns::cmp_node_chroma_matte_declare;
  ntype.draw_buttons = file_ns::node_composit_buts_chroma_matte;
  ntype.flag |= NODE_PREVIEW;
  node_type_init(&ntype, file_ns::node_composit_init_chroma_matte);
  node_type_storage(&ntype, "NodeChroma", node_free_standard_storage, node_copy_standard_storage);
  node_type_gpu(&ntype, file_ns::node_composite_gpu_chroma_matte);

  nodeRegisterType(&ntype);
}
