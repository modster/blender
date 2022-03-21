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

#include "GPU_material.h"

#include "NOD_compositor_execute.hh"

#include "node_composite_util.hh"

/* **************** SEPARATE RGBA ******************** */

namespace blender::nodes::node_composite_separate_rgba_cc {

static void cmp_node_seprgba_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Color>(N_("Image")).default_value({1.0f, 1.0f, 1.0f, 1.0f});
  b.add_output<decl::Float>(N_("R"));
  b.add_output<decl::Float>(N_("G"));
  b.add_output<decl::Float>(N_("B"));
  b.add_output<decl::Float>(N_("A"));
}

using namespace blender::viewport_compositor;

class SeparateRGBAGPUMaterialNode : public GPUMaterialNode {
 public:
  using GPUMaterialNode::GPUMaterialNode;

  void compile(GPUMaterial *material) override
  {
    GPUNodeStack *inputs = get_inputs_array();
    GPUNodeStack *outputs = get_outputs_array();

    GPU_stack_link(material, &node(), "node_composite_separate_rgba", inputs, outputs);
  }
};

static GPUMaterialNode *get_compositor_gpu_material_node(DNode node)
{
  return new SeparateRGBAGPUMaterialNode(node);
}

}  // namespace blender::nodes::node_composite_separate_rgba_cc

void register_node_type_cmp_seprgba()
{
  namespace file_ns = blender::nodes::node_composite_separate_rgba_cc;

  static bNodeType ntype;

  cmp_node_type_base(&ntype, CMP_NODE_SEPRGBA, "Separate RGBA", NODE_CLASS_CONVERTER);
  ntype.declare = file_ns::cmp_node_seprgba_declare;
  ntype.get_compositor_gpu_material_node = file_ns::get_compositor_gpu_material_node;

  nodeRegisterType(&ntype);
}

/* **************** COMBINE RGBA ******************** */

namespace blender::nodes::node_composite_combine_rgba_cc {

static void cmp_node_combrgba_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Float>(N_("R")).min(0.0f).max(1.0f);
  b.add_input<decl::Float>(N_("G")).min(0.0f).max(1.0f);
  b.add_input<decl::Float>(N_("B")).min(0.0f).max(1.0f);
  b.add_input<decl::Float>(N_("A")).default_value(1.0f).min(0.0f).max(1.0f);
  b.add_output<decl::Color>(N_("Image"));
}

using namespace blender::viewport_compositor;

class CombineRGBAGPUMaterialNode : public GPUMaterialNode {
 public:
  using GPUMaterialNode::GPUMaterialNode;

  void compile(GPUMaterial *material) override
  {
    GPUNodeStack *inputs = get_inputs_array();
    GPUNodeStack *outputs = get_outputs_array();

    GPU_stack_link(material, &node(), "node_composite_combine_rgba", inputs, outputs);
  }
};

static GPUMaterialNode *get_compositor_gpu_material_node(DNode node)
{
  return new CombineRGBAGPUMaterialNode(node);
}

}  // namespace blender::nodes::node_composite_combine_rgba_cc

void register_node_type_cmp_combrgba()
{
  namespace file_ns = blender::nodes::node_composite_combine_rgba_cc;

  static bNodeType ntype;

  cmp_node_type_base(&ntype, CMP_NODE_COMBRGBA, "Combine RGBA", NODE_CLASS_CONVERTER);
  ntype.declare = file_ns::cmp_node_combrgba_declare;
  ntype.get_compositor_gpu_material_node = file_ns::get_compositor_gpu_material_node;

  nodeRegisterType(&ntype);
}
