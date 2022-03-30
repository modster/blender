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

#include "BLI_assert.h"

#include "DNA_material_types.h"

#include "GPU_material.h"

#include "NOD_compositor_execute.hh"

#include "node_composite_util.hh"

/* **************** MIX RGB ******************** */

namespace blender::nodes::node_composite_mixrgb_cc {

static void cmp_node_mixrgb_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Float>(N_("Fac"))
      .default_value(1.0f)
      .min(0.0f)
      .max(1.0f)
      .subtype(PROP_FACTOR)
      .compositor_domain_priority(2);
  b.add_input<decl::Color>(N_("Image"))
      .default_value({1.0f, 1.0f, 1.0f, 1.0f})
      .compositor_domain_priority(0);
  b.add_input<decl::Color>(N_("Image"), "Image_001")
      .default_value({1.0f, 1.0f, 1.0f, 1.0f})
      .compositor_domain_priority(1);
  b.add_output<decl::Color>(N_("Image"));
}

using namespace blender::viewport_compositor;

class MixRGBGPUMaterialNode : public GPUMaterialNode {
 public:
  using GPUMaterialNode::GPUMaterialNode;

  void compile(GPUMaterial *material) override
  {
    GPUNodeStack *inputs = get_inputs_array();
    GPUNodeStack *outputs = get_outputs_array();

    GPU_stack_link(material, &node(), get_shader_function_name(), inputs, outputs);

    if (get_use_alpha()) {
      GPU_link(material, "multiply_by_alpha", inputs[0].link, inputs[1].link, &inputs[0].link);
    }

    if (!get_should_clamp()) {
      return;
    }

    const float min[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const float max[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    GPU_link(material,
             "clamp_color",
             outputs[0].link,
             GPU_constant(min),
             GPU_constant(max),
             &outputs[0].link);
  }

  int get_mode()
  {
    return node().custom1;
  }

  const char *get_shader_function_name()
  {
    switch (get_mode()) {
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

    BLI_assert_unreachable();
    return nullptr;
  }

  bool get_use_alpha()
  {
    return node().custom2 & SHD_MIXRGB_USE_ALPHA;
  }

  bool get_should_clamp()
  {
    return node().custom2 & SHD_MIXRGB_CLAMP;
  }
};

static GPUMaterialNode *get_compositor_gpu_material_node(DNode node)
{
  return new MixRGBGPUMaterialNode(node);
}

}  // namespace blender::nodes::node_composite_mixrgb_cc

void register_node_type_cmp_mix_rgb()
{
  namespace file_ns = blender::nodes::node_composite_mixrgb_cc;

  static bNodeType ntype;

  cmp_node_type_base(&ntype, CMP_NODE_MIX_RGB, "Mix", NODE_CLASS_OP_COLOR);
  ntype.flag |= NODE_PREVIEW;
  ntype.declare = file_ns::cmp_node_mixrgb_declare;
  ntype.labelfunc = node_blend_label;
  ntype.get_compositor_gpu_material_node = file_ns::get_compositor_gpu_material_node;

  nodeRegisterType(&ntype);
}
