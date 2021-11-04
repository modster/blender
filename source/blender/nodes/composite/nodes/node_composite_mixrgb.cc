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

#include "DNA_material_types.h"
#include "node_composite_util.hh"

/* **************** MIX RGB ******************** */

namespace blender::nodes {

static void cmp_node_mixrgb_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Float>("Fac").default_value(1.0f).min(0.0f).max(1.0f).subtype(PROP_FACTOR);
  b.add_input<decl::Color>("Image").default_value({1.0f, 1.0f, 1.0f, 1.0f});
  b.add_input<decl::Color>("Image", "Image_001").default_value({1.0f, 1.0f, 1.0f, 1.0f});
  b.add_output<decl::Color>("Image");
}

}  // namespace blender::nodes

static const char *gpu_shader_get_name(int mode)
{
  switch (mode) {
    case MA_RAMP_BLEND:
      return "mix_blend";
    case MA_RAMP_ADD:
      return "mix_add";
    case MA_RAMP_MULT:
      return "mix_mult";
    case MA_RAMP_SUB:
      return "mix_sub";
    case MA_RAMP_SCREEN:
      return "mix_screen";
    case MA_RAMP_DIV:
      return "mix_div";
    case MA_RAMP_DIFF:
      return "mix_diff";
    case MA_RAMP_DARK:
      return "mix_dark";
    case MA_RAMP_LIGHT:
      return "mix_light";
    case MA_RAMP_OVERLAY:
      return "mix_overlay";
    case MA_RAMP_DODGE:
      return "mix_dodge";
    case MA_RAMP_BURN:
      return "mix_burn";
    case MA_RAMP_HUE:
      return "mix_hue";
    case MA_RAMP_SAT:
      return "mix_sat";
    case MA_RAMP_VAL:
      return "mix_val";
    case MA_RAMP_COLOR:
      return "mix_color";
    case MA_RAMP_SOFT:
      return "mix_soft";
    case MA_RAMP_LINEAR:
      return "mix_linear";
  }

  return nullptr;
}

static int node_composite_gpu_mix_rgb(GPUMaterial *mat,
                                      bNode *node,
                                      bNodeExecData *UNUSED(execdata),
                                      GPUNodeStack *in,
                                      GPUNodeStack *out)
{
  const char *name = gpu_shader_get_name(node->custom1);
  if (name == nullptr) {
    return 0;
  }

  bool valid = GPU_stack_link(mat, node, name, in, out);
  if (!valid) {
    return 0;
  }

  if (node->custom2 & SHD_MIXRGB_USE_ALPHA) {
    bool valid = GPU_link(mat, "multiply_by_alpha", in[0].link, in[1].link, &in[0].link);
    if (!valid) {
      return 0;
    }
  }

  if (node->custom2 & SHD_MIXRGB_CLAMP) {
    const float min[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const float max[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    valid = GPU_link(
        mat, "clamp_color", out[0].link, GPU_constant(min), GPU_constant(max), &out[0].link);
    if (!valid) {
      return 0;
    }
  }

  return 1;
}

/* custom1 = mix type */
void register_node_type_cmp_mix_rgb(void)
{
  static bNodeType ntype;

  cmp_node_type_base(&ntype, CMP_NODE_MIX_RGB, "Mix", NODE_CLASS_OP_COLOR, NODE_PREVIEW);
  ntype.declare = blender::nodes::cmp_node_mixrgb_declare;
  node_type_label(&ntype, node_blend_label);
  node_type_gpu(&ntype, node_composite_gpu_mix_rgb);

  nodeRegisterType(&ntype);
}
