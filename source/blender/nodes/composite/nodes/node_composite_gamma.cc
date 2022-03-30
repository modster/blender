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

/* **************** Gamma Tools  ******************** */

namespace blender::nodes::node_composite_gamma_cc {

static void cmp_node_gamma_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Color>(N_("Image"))
      .default_value({1.0f, 1.0f, 1.0f, 1.0f})
      .compositor_domain_priority(0);
  b.add_input<decl::Float>(N_("Gamma"))
      .default_value(1.0f)
      .min(0.001f)
      .max(10.0f)
      .subtype(PROP_UNSIGNED)
      .compositor_domain_priority(1);
  b.add_output<decl::Color>(N_("Image"));
}

using namespace blender::viewport_compositor;

class GammaGPUMaterialNode : public GPUMaterialNode {
 public:
  using GPUMaterialNode::GPUMaterialNode;

  void compile(GPUMaterial *material) override
  {
    GPUNodeStack *inputs = get_inputs_array();
    GPUNodeStack *outputs = get_outputs_array();

    GPU_stack_link(material, &node(), "node_composite_gamma", inputs, outputs);
  }
};

static GPUMaterialNode *get_compositor_gpu_material_node(DNode node)
{
  return new GammaGPUMaterialNode(node);
}

}  // namespace blender::nodes::node_composite_gamma_cc

void register_node_type_cmp_gamma()
{
  namespace file_ns = blender::nodes::node_composite_gamma_cc;

  static bNodeType ntype;

  cmp_node_type_base(&ntype, CMP_NODE_GAMMA, "Gamma", NODE_CLASS_OP_COLOR);
  ntype.declare = file_ns::cmp_node_gamma_declare;
  ntype.get_compositor_gpu_material_node = file_ns::get_compositor_gpu_material_node;

  nodeRegisterType(&ntype);
}
