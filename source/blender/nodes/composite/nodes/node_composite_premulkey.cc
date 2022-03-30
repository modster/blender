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

#include "GPU_material.h"

#include "NOD_compositor_execute.hh"

#include "node_composite_util.hh"

/* **************** Premul and Key Alpha Convert ******************** */

namespace blender::nodes::node_composite_premulkey_cc {

static void cmp_node_premulkey_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Color>(N_("Image"))
      .default_value({1.0f, 1.0f, 1.0f, 1.0f})
      .compositor_domain_priority(0);
  b.add_output<decl::Color>(N_("Image"));
}

static void node_composit_buts_premulkey(uiLayout *layout, bContext *UNUSED(C), PointerRNA *ptr)
{
  uiItemR(layout, ptr, "mapping", UI_ITEM_R_SPLIT_EMPTY_NAME, "", ICON_NONE);
}

using namespace blender::viewport_compositor;

class AlphaConvertGPUMaterialNode : public GPUMaterialNode {
 public:
  using GPUMaterialNode::GPUMaterialNode;

  void compile(GPUMaterial *material) override
  {
    GPUNodeStack *inputs = get_inputs_array();
    GPUNodeStack *outputs = get_outputs_array();

    if (get_mode() == 0) {
      GPU_stack_link(material, &node(), "color_alpha_premultiply", inputs, outputs);
      return;
    }

    GPU_stack_link(material, &node(), "color_alpha_unpremultiply", inputs, outputs);
  }

  /* 0 -> Premultiply Alpha.
   * 1 -> Unpremultiply Alpha. */
  int get_mode()
  {
    return node().custom1;
  }
};

static GPUMaterialNode *get_compositor_gpu_material_node(DNode node)
{
  return new AlphaConvertGPUMaterialNode(node);
}

}  // namespace blender::nodes::node_composite_premulkey_cc

void register_node_type_cmp_premulkey()
{
  namespace file_ns = blender::nodes::node_composite_premulkey_cc;

  static bNodeType ntype;

  cmp_node_type_base(&ntype, CMP_NODE_PREMULKEY, "Alpha Convert", NODE_CLASS_CONVERTER);
  ntype.declare = file_ns::cmp_node_premulkey_declare;
  ntype.draw_buttons = file_ns::node_composit_buts_premulkey;
  ntype.get_compositor_gpu_material_node = file_ns::get_compositor_gpu_material_node;

  nodeRegisterType(&ntype);
}
