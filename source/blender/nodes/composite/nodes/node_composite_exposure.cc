/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2020 Blender Foundation. All rights reserved. */

/** \file
 * \ingroup cmpnodes
 */

#include "GPU_material.h"

#include "VPC_gpu_material_node.hh"

#include "node_composite_util.hh"

/* **************** Exposure ******************** */

namespace blender::nodes::node_composite_exposure_cc {

static void cmp_node_exposure_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Color>(N_("Image"))
      .default_value({1.0f, 1.0f, 1.0f, 1.0f})
      .compositor_domain_priority(0);
  b.add_input<decl::Float>(N_("Exposure")).min(-10.0f).max(10.0f).compositor_domain_priority(1);
  b.add_output<decl::Color>(N_("Image"));
}

using namespace blender::viewport_compositor;

class ExposureGPUMaterialNode : public GPUMaterialNode {
 public:
  using GPUMaterialNode::GPUMaterialNode;

  void compile(GPUMaterial *material) override
  {
    GPUNodeStack *inputs = get_inputs_array();
    GPUNodeStack *outputs = get_outputs_array();

    GPU_stack_link(material, &bnode(), "node_composite_exposure", inputs, outputs);
  }
};

static GPUMaterialNode *get_compositor_gpu_material_node(DNode node)
{
  return new ExposureGPUMaterialNode(node);
}

}  // namespace blender::nodes::node_composite_exposure_cc

void register_node_type_cmp_exposure()
{
  namespace file_ns = blender::nodes::node_composite_exposure_cc;

  static bNodeType ntype;

  cmp_node_type_base(&ntype, CMP_NODE_EXPOSURE, "Exposure", NODE_CLASS_OP_COLOR);
  ntype.declare = file_ns::cmp_node_exposure_declare;
  ntype.get_compositor_gpu_material_node = file_ns::get_compositor_gpu_material_node;

  nodeRegisterType(&ntype);
}
