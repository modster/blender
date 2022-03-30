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

#include "BKE_colortools.h"

#include "GPU_material.h"

#include "NOD_compositor_execute.hh"

#include "node_composite_util.hh"

namespace blender::nodes::node_composite_huecorrect_cc {

static void cmp_node_huecorrect_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Float>(N_("Fac"))
      .default_value(1.0f)
      .min(0.0f)
      .max(1.0f)
      .subtype(PROP_FACTOR)
      .compositor_domain_priority(1);
  b.add_input<decl::Color>(N_("Image"))
      .default_value({1.0f, 1.0f, 1.0f, 1.0f})
      .compositor_domain_priority(0);
  b.add_output<decl::Color>(N_("Image"));
}

static void node_composit_init_huecorrect(bNodeTree *UNUSED(ntree), bNode *node)
{
  node->storage = BKE_curvemapping_add(1, 0.0f, 0.0f, 1.0f, 1.0f);

  CurveMapping *cumapping = (CurveMapping *)node->storage;

  cumapping->preset = CURVE_PRESET_MID9;

  for (int c = 0; c < 3; c++) {
    CurveMap *cuma = &cumapping->cm[c];
    BKE_curvemap_reset(cuma, &cumapping->clipr, cumapping->preset, CURVEMAP_SLOPE_POSITIVE);
  }

  /* default to showing Saturation */
  cumapping->cur = 1;
}

using namespace blender::viewport_compositor;

class HueCorrectGPUMaterialNode : public GPUMaterialNode {
 public:
  using GPUMaterialNode::GPUMaterialNode;

  void compile(GPUMaterial *material) override
  {
    GPUNodeStack *inputs = get_inputs_array();
    GPUNodeStack *outputs = get_outputs_array();

    CurveMapping *curve_mapping = get_curve_mapping();

    BKE_curvemapping_init(curve_mapping);
    float *band_values;
    int band_size;
    BKE_curvemapping_table_RGBA(curve_mapping, &band_values, &band_size);
    float band_layer;
    GPUNodeLink *band_texture = GPU_color_band(material, band_size, band_values, &band_layer);

    float range_minimums[CM_TOT];
    BKE_curvemapping_get_range_minimums(curve_mapping, range_minimums);
    float range_dividers[CM_TOT];
    BKE_curvemapping_compute_range_dividers(curve_mapping, range_dividers);

    GPU_stack_link(material,
                   &node(),
                   "node_composite_hue_correct",
                   inputs,
                   outputs,
                   band_texture,
                   GPU_constant(&band_layer),
                   GPU_uniform(range_minimums),
                   GPU_uniform(range_dividers));
  }

  CurveMapping *get_curve_mapping()
  {
    return static_cast<CurveMapping *>(node().storage);
  }
};

static GPUMaterialNode *get_compositor_gpu_material_node(DNode node)
{
  return new HueCorrectGPUMaterialNode(node);
}

}  // namespace blender::nodes::node_composite_huecorrect_cc

void register_node_type_cmp_huecorrect()
{
  namespace file_ns = blender::nodes::node_composite_huecorrect_cc;

  static bNodeType ntype;

  cmp_node_type_base(&ntype, CMP_NODE_HUECORRECT, "Hue Correct", NODE_CLASS_OP_COLOR);
  ntype.declare = file_ns::cmp_node_huecorrect_declare;
  node_type_size(&ntype, 320, 140, 500);
  node_type_init(&ntype, file_ns::node_composit_init_huecorrect);
  node_type_storage(&ntype, "CurveMapping", node_free_curves, node_copy_curves);
  ntype.get_compositor_gpu_material_node = file_ns::get_compositor_gpu_material_node;

  nodeRegisterType(&ntype);
}
