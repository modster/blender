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

#include "IMB_colormanagement.h"

#include "BKE_colorband.h"

#include "GPU_material.h"

#include "VPC_compositor_execute.hh"

#include "node_composite_util.hh"

/* **************** VALTORGB ******************** */

namespace blender::nodes::node_composite_color_ramp_cc {

static void cmp_node_valtorgb_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Float>(N_("Fac"))
      .default_value(0.5f)
      .min(0.0f)
      .max(1.0f)
      .subtype(PROP_FACTOR)
      .compositor_domain_priority(1);
  b.add_output<decl::Color>(N_("Image")).compositor_domain_priority(0);
  b.add_output<decl::Float>(N_("Alpha"));
}

static void node_composit_init_valtorgb(bNodeTree *UNUSED(ntree), bNode *node)
{
  node->storage = BKE_colorband_add(true);
}

using namespace blender::viewport_compositor;

class ColorRampGPUMaterialNode : public GPUMaterialNode {
 public:
  using GPUMaterialNode::GPUMaterialNode;

  void compile(GPUMaterial *material) override
  {
    GPUNodeStack *inputs = get_inputs_array();
    GPUNodeStack *outputs = get_outputs_array();

    struct ColorBand *color_band = get_color_band();

    /* Common / easy case optimization. */
    if ((color_band->tot <= 2) && (color_band->color_mode == COLBAND_BLEND_RGB)) {
      float mul_bias[2];
      switch (color_band->ipotype) {
        case COLBAND_INTERP_LINEAR:
          mul_bias[0] = 1.0f / (color_band->data[1].pos - color_band->data[0].pos);
          mul_bias[1] = -mul_bias[0] * color_band->data[0].pos;
          GPU_stack_link(material,
                         &node(),
                         "valtorgb_opti_linear",
                         inputs,
                         outputs,
                         GPU_uniform(mul_bias),
                         GPU_uniform(&color_band->data[0].r),
                         GPU_uniform(&color_band->data[1].r));
          return;
        case COLBAND_INTERP_CONSTANT:
          mul_bias[1] = max_ff(color_band->data[0].pos, color_band->data[1].pos);
          GPU_stack_link(material,
                         &node(),
                         "valtorgb_opti_constant",
                         inputs,
                         outputs,
                         GPU_uniform(&mul_bias[1]),
                         GPU_uniform(&color_band->data[0].r),
                         GPU_uniform(&color_band->data[1].r));
          return;
        case COLBAND_INTERP_EASE:
          mul_bias[0] = 1.0f / (color_band->data[1].pos - color_band->data[0].pos);
          mul_bias[1] = -mul_bias[0] * color_band->data[0].pos;
          GPU_stack_link(material,
                         &node(),
                         "valtorgb_opti_ease",
                         inputs,
                         outputs,
                         GPU_uniform(mul_bias),
                         GPU_uniform(&color_band->data[0].r),
                         GPU_uniform(&color_band->data[1].r));
          return;
        default:
          BLI_assert_unreachable();
          return;
      }
    }

    float *array, layer;
    int size;
    BKE_colorband_evaluate_table_rgba(color_band, &array, &size);
    GPUNodeLink *tex = GPU_color_band(material, size, array, &layer);

    if (color_band->ipotype == COLBAND_INTERP_CONSTANT) {
      GPU_stack_link(
          material, &node(), "valtorgb_nearest", inputs, outputs, tex, GPU_constant(&layer));
      return;
    }

    GPU_stack_link(material, &node(), "valtorgb", inputs, outputs, tex, GPU_constant(&layer));
  }

  struct ColorBand *get_color_band()
  {
    return static_cast<struct ColorBand *>(node().storage);
  }
};

static GPUMaterialNode *get_compositor_gpu_material_node(DNode node)
{
  return new ColorRampGPUMaterialNode(node);
}

}  // namespace blender::nodes::node_composite_color_ramp_cc

void register_node_type_cmp_valtorgb()
{
  namespace file_ns = blender::nodes::node_composite_color_ramp_cc;

  static bNodeType ntype;

  cmp_node_type_base(&ntype, CMP_NODE_VALTORGB, "ColorRamp", NODE_CLASS_CONVERTER);
  ntype.declare = file_ns::cmp_node_valtorgb_declare;
  node_type_size(&ntype, 240, 200, 320);
  node_type_init(&ntype, file_ns::node_composit_init_valtorgb);
  node_type_storage(&ntype, "ColorBand", node_free_standard_storage, node_copy_standard_storage);
  ntype.get_compositor_gpu_material_node = file_ns::get_compositor_gpu_material_node;

  nodeRegisterType(&ntype);
}

/* **************** RGBTOBW ******************** */

namespace blender::nodes::node_composite_rgb_to_bw_cc {

static void cmp_node_rgbtobw_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Color>(N_("Image"))
      .default_value({0.8f, 0.8f, 0.8f, 1.0f})
      .compositor_domain_priority(0);
  b.add_output<decl::Color>(N_("Val"));
}

using namespace blender::viewport_compositor;

class RGBToBWGPUMaterialNode : public GPUMaterialNode {
 public:
  using GPUMaterialNode::GPUMaterialNode;

  void compile(GPUMaterial *material) override
  {
    GPUNodeStack *inputs = get_inputs_array();
    GPUNodeStack *outputs = get_outputs_array();

    float luminance_coefficients[3];
    IMB_colormanagement_get_luminance_coefficients(luminance_coefficients);

    GPU_stack_link(material,
                   &node(),
                   "color_to_luminance",
                   inputs,
                   outputs,
                   GPU_constant(luminance_coefficients));
  }
};

static GPUMaterialNode *get_compositor_gpu_material_node(DNode node)
{
  return new RGBToBWGPUMaterialNode(node);
}

}  // namespace blender::nodes::node_composite_rgb_to_bw_cc

void register_node_type_cmp_rgbtobw()
{
  namespace file_ns = blender::nodes::node_composite_rgb_to_bw_cc;

  static bNodeType ntype;

  cmp_node_type_base(&ntype, CMP_NODE_RGBTOBW, "RGB to BW", NODE_CLASS_CONVERTER);
  ntype.declare = file_ns::cmp_node_rgbtobw_declare;
  node_type_size_preset(&ntype, NODE_SIZE_SMALL);
  ntype.get_compositor_gpu_material_node = file_ns::get_compositor_gpu_material_node;

  nodeRegisterType(&ntype);
}
