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

namespace blender::nodes {

static void cmp_node_huecorrect_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Float>("Fac").default_value(1.0f).min(0.0f).max(1.0f).subtype(PROP_FACTOR);
  b.add_input<decl::Color>("Image").default_value({1.0f, 1.0f, 1.0f, 1.0f});
  b.add_output<decl::Color>("Image");
}

}  // namespace blender::nodes

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

static int node_composite_gpu_huecorrect(GPUMaterial *mat,
                                         bNode *node,
                                         bNodeExecData *UNUSED(execdata),
                                         GPUNodeStack *in,
                                         GPUNodeStack *out)
{
  CurveMapping *curve_mapping = (CurveMapping *)node->storage;

  BKE_curvemapping_init(curve_mapping);
  float *band_values;
  int band_size;
  BKE_curvemapping_table_RGBA(curve_mapping, &band_values, &band_size);
  float band_layer;
  GPUNodeLink *band_texture = GPU_color_band(mat, band_size, band_values, &band_layer);

  float range_minimums[CM_TOT];
  BKE_curvemapping_get_range_minimums(curve_mapping, range_minimums);
  float range_dividers[CM_TOT];
  BKE_curvemapping_compute_range_dividers(curve_mapping, range_dividers);

  return GPU_stack_link(mat,
                        node,
                        "node_composite_hue_correct",
                        in,
                        out,
                        band_texture,
                        GPU_constant(&band_layer),
                        GPU_uniform(range_minimums),
                        GPU_uniform(range_dividers));
}

void register_node_type_cmp_huecorrect(void)
{
  static bNodeType ntype;

  cmp_node_type_base(&ntype, CMP_NODE_HUECORRECT, "Hue Correct", NODE_CLASS_OP_COLOR, 0);
  ntype.declare = blender::nodes::cmp_node_huecorrect_declare;
  node_type_size(&ntype, 320, 140, 500);
  node_type_init(&ntype, node_composit_init_huecorrect);
  node_type_storage(&ntype, "CurveMapping", node_free_curves, node_copy_curves);
  node_type_gpu(&ntype, node_composite_gpu_huecorrect);

  nodeRegisterType(&ntype);
}
