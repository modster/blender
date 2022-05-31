/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2021 Blender Foundation. All rights reserved. */

/** \file
 * \ingroup cmpnodes
 */

#include "GPU_material.h"

#include "COM_gpu_material_node.hh"

#include "node_composite_util.hh"

/* **************** SEPARATE XYZ ******************** */

namespace blender::nodes::node_composite_separate_xyz_cc {

static void cmp_node_separate_xyz_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Vector>("Vector").min(-10000.0f).max(10000.0f);
  b.add_output<decl::Float>("X");
  b.add_output<decl::Float>("Y");
  b.add_output<decl::Float>("Z");
}

using namespace blender::realtime_compositor;

class SeparateXYZGPUMaterialNode : public GPUMaterialNode {
 public:
  using GPUMaterialNode::GPUMaterialNode;

  void compile(GPUMaterial *material) override
  {
    GPUNodeStack *inputs = get_inputs_array();
    GPUNodeStack *outputs = get_outputs_array();

    GPU_stack_link(material, &bnode(), "node_composite_separate_xyz", inputs, outputs);
  }
};

static GPUMaterialNode *get_compositor_gpu_material_node(DNode node)
{
  return new SeparateXYZGPUMaterialNode(node);
}

}  // namespace blender::nodes::node_composite_separate_xyz_cc

void register_node_type_cmp_separate_xyz()
{
  namespace file_ns = blender::nodes::node_composite_separate_xyz_cc;

  static bNodeType ntype;

  cmp_node_type_base(&ntype, CMP_NODE_SEPARATE_XYZ, "Separate XYZ", NODE_CLASS_CONVERTER);
  ntype.declare = file_ns::cmp_node_separate_xyz_declare;
  ntype.get_compositor_gpu_material_node = file_ns::get_compositor_gpu_material_node;

  nodeRegisterType(&ntype);
}

/* **************** COMBINE XYZ ******************** */

namespace blender::nodes::node_composite_combine_xyz_cc {

static void cmp_node_combine_xyz_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Float>("X").min(-10000.0f).max(10000.0f);
  b.add_input<decl::Float>("Y").min(-10000.0f).max(10000.0f);
  b.add_input<decl::Float>("Z").min(-10000.0f).max(10000.0f);
  b.add_output<decl::Vector>("Vector");
}

using namespace blender::realtime_compositor;

class CombineXYZGPUMaterialNode : public GPUMaterialNode {
 public:
  using GPUMaterialNode::GPUMaterialNode;

  void compile(GPUMaterial *material) override
  {
    GPUNodeStack *inputs = get_inputs_array();
    GPUNodeStack *outputs = get_outputs_array();

    GPU_stack_link(material, &bnode(), "node_composite_combine_xyz", inputs, outputs);
  }
};

static GPUMaterialNode *get_compositor_gpu_material_node(DNode node)
{
  return new CombineXYZGPUMaterialNode(node);
}

}  // namespace blender::nodes::node_composite_combine_xyz_cc

void register_node_type_cmp_combine_xyz()
{
  namespace file_ns = blender::nodes::node_composite_combine_xyz_cc;

  static bNodeType ntype;

  cmp_node_type_base(&ntype, CMP_NODE_COMBINE_XYZ, "Combine XYZ", NODE_CLASS_CONVERTER);
  ntype.declare = file_ns::cmp_node_combine_xyz_declare;
  ntype.get_compositor_gpu_material_node = file_ns::get_compositor_gpu_material_node;

  nodeRegisterType(&ntype);
}
