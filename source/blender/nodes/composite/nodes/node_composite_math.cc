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

#include "NOD_math_functions.hh"

/* **************** SCALAR MATH ******************** */

namespace blender::nodes::node_composite_math_cc {

static void cmp_node_math_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Float>(N_("Value")).default_value(0.5f).min(-10000.0f).max(10000.0f);
  b.add_input<decl::Float>(N_("Value"), "Value_001")
      .default_value(0.5f)
      .min(-10000.0f)
      .max(10000.0f);
  b.add_input<decl::Float>(N_("Value"), "Value_002")
      .default_value(0.5f)
      .min(-10000.0f)
      .max(10000.0f);
  b.add_output<decl::Float>(N_("Value"));
}

static const char *gpu_shader_get_name(int mode)
{
  const blender::nodes::FloatMathOperationInfo *info =
      blender::nodes::get_float_math_operation_info(mode);
  if (!info) {
    return nullptr;
  }
  if (info->shader_name.is_empty()) {
    return nullptr;
  }
  return info->shader_name.c_str();
}

static int node_composite_gpu_math(GPUMaterial *mat,
                                   bNode *node,
                                   bNodeExecData *UNUSED(execdata),
                                   GPUNodeStack *in,
                                   GPUNodeStack *out)
{
  const char *name = gpu_shader_get_name(node->custom1);
  if (name == nullptr) {
    return 0;
  }

  int valid = GPU_stack_link(mat, node, name, in, out);
  if (!valid) {
    return 0;
  }

  if (node->custom2 & SHD_MATH_CLAMP) {
    const float min = 0.0f;
    const float max = 1.0f;
    return GPU_link(
        mat, "clamp_value", out[0].link, GPU_constant(&min), GPU_constant(&max), &out[0].link);
  }

  return 1;
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
  node_type_gpu(&ntype, file_ns::node_composite_gpu_math);

  nodeRegisterType(&ntype);
}
