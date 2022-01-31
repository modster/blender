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

#include "node_composite_util.hh"

/* **************** SEPARATE YUVA ******************** */

namespace blender::nodes::node_composite_sepcomb_yuva_cc {

static void cmp_node_sepyuva_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Color>(N_("Image")).default_value({1.0f, 1.0f, 1.0f, 1.0f});
  b.add_output<decl::Float>(N_("Y"));
  b.add_output<decl::Float>(N_("U"));
  b.add_output<decl::Float>(N_("V"));
  b.add_output<decl::Float>(N_("A"));
}

static int node_composite_gpu_sepyuva(GPUMaterial *mat,
                                      bNode *node,
                                      bNodeExecData *UNUSED(execdata),
                                      GPUNodeStack *in,
                                      GPUNodeStack *out)
{
  return GPU_stack_link(mat, node, "node_composite_separate_yuva_itu_709", in, out);
}

}  // namespace blender::nodes::node_composite_sepcomb_yuva_cc

void register_node_type_cmp_sepyuva()
{
  namespace file_ns = blender::nodes::node_composite_sepcomb_yuva_cc;

  static bNodeType ntype;

  cmp_node_type_base(&ntype, CMP_NODE_SEPYUVA, "Separate YUVA", NODE_CLASS_CONVERTER);
  ntype.declare = file_ns::cmp_node_sepyuva_declare;
  node_type_gpu(&ntype, file_ns::node_composite_gpu_sepyuva);

  nodeRegisterType(&ntype);
}

/* **************** COMBINE YUVA ******************** */

namespace blender::nodes::node_composite_sepcomb_yuva_cc {

static void cmp_node_combyuva_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Float>(N_("Y")).min(0.0f).max(1.0f);
  b.add_input<decl::Float>(N_("U")).min(0.0f).max(1.0f);
  b.add_input<decl::Float>(N_("V")).min(0.0f).max(1.0f);
  b.add_input<decl::Float>(N_("A")).default_value(1.0f).min(0.0f).max(1.0f);
  b.add_output<decl::Color>(N_("Image"));
}

static int node_composite_gpu_combyuva(GPUMaterial *mat,
                                       bNode *node,
                                       bNodeExecData *UNUSED(execdata),
                                       GPUNodeStack *in,
                                       GPUNodeStack *out)
{
  return GPU_stack_link(mat, node, "node_composite_combine_yuva_itu_709", in, out);
}

}  // namespace blender::nodes::node_composite_sepcomb_yuva_cc

void register_node_type_cmp_combyuva()
{
  namespace file_ns = blender::nodes::node_composite_sepcomb_yuva_cc;

  static bNodeType ntype;

  cmp_node_type_base(&ntype, CMP_NODE_COMBYUVA, "Combine YUVA", NODE_CLASS_CONVERTER);
  ntype.declare = file_ns::cmp_node_combyuva_declare;
  node_type_gpu(&ntype, file_ns::node_composite_gpu_combyuva);

  nodeRegisterType(&ntype);
}
