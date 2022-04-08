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

#include "VPC_compositor_execute.hh"

#include "NOD_math_functions.hh"

#include "node_composite_util.hh"

/* **************** SCALAR MATH ******************** */

namespace blender::nodes::node_composite_math_cc {

static void cmp_node_math_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Float>(N_("Value"))
      .default_value(0.5f)
      .min(-10000.0f)
      .max(10000.0f)
      .compositor_domain_priority(0);
  b.add_input<decl::Float>(N_("Value"), "Value_001")
      .default_value(0.5f)
      .min(-10000.0f)
      .max(10000.0f)
      .compositor_domain_priority(1);
  b.add_input<decl::Float>(N_("Value"), "Value_002")
      .default_value(0.5f)
      .min(-10000.0f)
      .max(10000.0f)
      .compositor_domain_priority(2);
  b.add_output<decl::Float>(N_("Value"));
}

using namespace blender::viewport_compositor;

class MathGPUMaterialNode : public GPUMaterialNode {
 public:
  using GPUMaterialNode::GPUMaterialNode;

  void compile(GPUMaterial *material) override
  {
    GPUNodeStack *inputs = get_inputs_array();
    GPUNodeStack *outputs = get_outputs_array();

    GPU_stack_link(material, &node(), get_shader_function_name(), inputs, outputs);

    if (!get_should_clamp()) {
      return;
    }

    const float min = 0.0f;
    const float max = 1.0f;
    GPU_link(material,
             "clamp_value",
             outputs[0].link,
             GPU_constant(&min),
             GPU_constant(&max),
             &outputs[0].link);
  }

  int get_mode()
  {
    return node().custom1;
  }

  const char *get_shader_function_name()
  {
    return get_float_math_operation_info(get_mode())->shader_name.c_str();
  }

  bool get_should_clamp()
  {
    return node().custom2 & SHD_MATH_CLAMP;
  }
};

static GPUMaterialNode *get_compositor_gpu_material_node(DNode node)
{
  return new MathGPUMaterialNode(node);
}

}  // namespace blender::nodes::node_composite_math_cc

void register_node_type_cmp_math()
{
  namespace file_ns = blender::nodes::node_composite_math_cc;

  static bNodeType ntype;

  cmp_node_type_base(&ntype, CMP_NODE_MATH, "Math", NODE_CLASS_CONVERTER);
  ntype.declare = file_ns::cmp_node_math_declare;
  ntype.labelfunc = node_math_label;
  node_type_update(&ntype, node_math_update);
  ntype.get_compositor_gpu_material_node = file_ns::get_compositor_gpu_material_node;

  nodeRegisterType(&ntype);
}
