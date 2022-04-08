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

#include "VPC_compositor_execute.hh"

#include "node_composite_util.hh"

/* **************** Map Range ******************** */

namespace blender::nodes::node_composite_map_range_cc {

static void cmp_node_map_range_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Float>(N_("Value"))
      .default_value(1.0f)
      .min(0.0f)
      .max(1.0f)
      .compositor_domain_priority(0);
  b.add_input<decl::Float>(N_("From Min"))
      .default_value(0.0f)
      .min(-10000.0f)
      .max(10000.0f)
      .compositor_domain_priority(1);
  b.add_input<decl::Float>(N_("From Max"))
      .default_value(0.0f)
      .min(-10000.0f)
      .max(10000.0f)
      .compositor_domain_priority(2);
  b.add_input<decl::Float>(N_("To Min"))
      .default_value(0.0f)
      .min(-10000.0f)
      .max(10000.0f)
      .compositor_domain_priority(3);
  b.add_input<decl::Float>(N_("To Max"))
      .default_value(0.0f)
      .min(-10000.0f)
      .max(10000.0f)
      .compositor_domain_priority(4);
  b.add_output<decl::Float>(N_("Value"));
}

static void node_composit_buts_map_range(uiLayout *layout, bContext *UNUSED(C), PointerRNA *ptr)
{
  uiLayout *col;

  col = uiLayoutColumn(layout, true);
  uiItemR(col, ptr, "use_clamp", UI_ITEM_R_SPLIT_EMPTY_NAME, nullptr, ICON_NONE);
}

using namespace blender::viewport_compositor;

class MapRangeGPUMaterialNode : public GPUMaterialNode {
 public:
  using GPUMaterialNode::GPUMaterialNode;

  void compile(GPUMaterial *material) override
  {
    GPUNodeStack *inputs = get_inputs_array();
    GPUNodeStack *outputs = get_outputs_array();

    const float should_clamp = get_should_clamp();

    GPU_stack_link(material,
                   &node(),
                   "node_composite_map_range",
                   inputs,
                   outputs,
                   GPU_constant(&should_clamp));
  }

  bool get_should_clamp()
  {
    return node().custom1;
  }
};

static GPUMaterialNode *get_compositor_gpu_material_node(DNode node)
{
  return new MapRangeGPUMaterialNode(node);
}

}  // namespace blender::nodes::node_composite_map_range_cc

void register_node_type_cmp_map_range()
{
  namespace file_ns = blender::nodes::node_composite_map_range_cc;

  static bNodeType ntype;

  cmp_node_type_base(&ntype, CMP_NODE_MAP_RANGE, "Map Range", NODE_CLASS_OP_VECTOR);
  ntype.declare = file_ns::cmp_node_map_range_declare;
  ntype.draw_buttons = file_ns::node_composit_buts_map_range;
  ntype.get_compositor_gpu_material_node = file_ns::get_compositor_gpu_material_node;

  nodeRegisterType(&ntype);
}
