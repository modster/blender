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

#include "node_composite_util.hh"

/* **************** NORMAL  ******************** */

namespace blender::nodes::node_composite_normal_cc {

static void cmp_node_normal_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Vector>(N_("Normal"))
      .default_value({1.0f, 1.0f, 1.0f})
      .min(-1.0f)
      .max(1.0f)
      .subtype(PROP_DIRECTION)
      .compositor_domain_priority(0);
  b.add_output<decl::Vector>(N_("Normal"));
  b.add_output<decl::Float>(N_("Dot"));
}

using namespace blender::viewport_compositor;

class NormalGPUMaterialNode : public GPUMaterialNode {
 public:
  using GPUMaterialNode::GPUMaterialNode;

  void compile(GPUMaterial *material) override
  {
    GPUNodeStack *inputs = get_inputs_array();
    GPUNodeStack *outputs = get_outputs_array();

    GPU_stack_link(material,
                   &node(),
                   "node_composite_normal",
                   inputs,
                   outputs,
                   GPU_uniform(get_vector_value()));
  }

  /* The vector value is stored in the default value of the output socket. */
  float *get_vector_value()
  {
    return get_outputs_array()[0].vec;
  }
};

static GPUMaterialNode *get_compositor_gpu_material_node(DNode node)
{
  return new NormalGPUMaterialNode(node);
}

}  // namespace blender::nodes::node_composite_normal_cc

void register_node_type_cmp_normal()
{
  namespace file_ns = blender::nodes::node_composite_normal_cc;

  static bNodeType ntype;

  cmp_node_type_base(&ntype, CMP_NODE_NORMAL, "Normal", NODE_CLASS_OP_VECTOR);
  ntype.declare = file_ns::cmp_node_normal_declare;
  ntype.get_compositor_gpu_material_node = file_ns::get_compositor_gpu_material_node;

  nodeRegisterType(&ntype);
}
