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

/* **************** Bright and Contrast  ******************** */

namespace blender::nodes::node_composite_brightness_cc {

static void cmp_node_brightcontrast_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Color>(N_("Image"))
      .default_value({1.0f, 1.0f, 1.0f, 1.0f})
      .compositor_domain_priority(0);
  b.add_input<decl::Float>(N_("Bright")).min(-100.0f).max(100.0f).compositor_domain_priority(1);
  b.add_input<decl::Float>(N_("Contrast")).min(-100.0f).max(100.0f).compositor_domain_priority(2);
  b.add_output<decl::Color>(N_("Image"));
}

static void node_composit_init_brightcontrast(bNodeTree *UNUSED(ntree), bNode *node)
{
  node->custom1 = 1;
}

static void node_composit_buts_brightcontrast(uiLayout *layout,
                                              bContext *UNUSED(C),
                                              PointerRNA *ptr)
{
  uiItemR(layout, ptr, "use_premultiply", UI_ITEM_R_SPLIT_EMPTY_NAME, nullptr, ICON_NONE);
}

using namespace blender::viewport_compositor;

class BrightContrastGPUMaterialNode : public GPUMaterialNode {
 public:
  using GPUMaterialNode::GPUMaterialNode;

  void compile(GPUMaterial *material) override
  {
    GPUNodeStack *inputs = get_inputs_array();
    GPUNodeStack *outputs = get_outputs_array();

    const float use_premultiply = get_use_premultiply();

    GPU_stack_link(material,
                   &node(),
                   "node_composite_bright_contrast",
                   inputs,
                   outputs,
                   GPU_constant(&use_premultiply));
  }

  bool get_use_premultiply()
  {
    return node().custom1;
  }
};

static GPUMaterialNode *get_compositor_gpu_material_node(DNode node)
{
  return new BrightContrastGPUMaterialNode(node);
}

}  // namespace blender::nodes::node_composite_brightness_cc

void register_node_type_cmp_brightcontrast()
{
  namespace file_ns = blender::nodes::node_composite_brightness_cc;

  static bNodeType ntype;

  cmp_node_type_base(&ntype, CMP_NODE_BRIGHTCONTRAST, "Bright/Contrast", NODE_CLASS_OP_COLOR);
  ntype.declare = file_ns::cmp_node_brightcontrast_declare;
  ntype.draw_buttons = file_ns::node_composit_buts_brightcontrast;
  node_type_init(&ntype, file_ns::node_composit_init_brightcontrast);
  ntype.get_compositor_gpu_material_node = file_ns::get_compositor_gpu_material_node;

  nodeRegisterType(&ntype);
}
