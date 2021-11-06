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

/* **************** CURVE Time  ******************** */

namespace blender::nodes {

static void cmp_node_time_declare(NodeDeclarationBuilder &b)
{
  b.add_output<decl::Float>("Fac");
}

}  // namespace blender::nodes

/* custom1 = start_frame, custom2 = end_frame */
static void node_composit_init_curves_time(bNodeTree *UNUSED(ntree), bNode *node)
{
  node->custom1 = 1;
  node->custom2 = 250;
  node->storage = BKE_curvemapping_add(1, 0.0f, 0.0f, 1.0f, 1.0f);
}

void register_node_type_cmp_curve_time(void)
{
  static bNodeType ntype;

  cmp_node_type_base(&ntype, CMP_NODE_TIME, "Time", NODE_CLASS_INPUT, 0);
  ntype.declare = blender::nodes::cmp_node_time_declare;
  node_type_size(&ntype, 140, 100, 320);
  node_type_init(&ntype, node_composit_init_curves_time);
  node_type_storage(&ntype, "CurveMapping", node_free_curves, node_copy_curves);

  nodeRegisterType(&ntype);
}

/* **************** CURVE VEC  ******************** */
static bNodeSocketTemplate cmp_node_curve_vec_in[] = {
    {SOCK_VECTOR, N_("Vector"), 0.0f, 0.0f, 0.0f, 1.0f, -1.0f, 1.0f, PROP_NONE},
    {-1, ""},
};

static bNodeSocketTemplate cmp_node_curve_vec_out[] = {
    {SOCK_VECTOR, N_("Vector")},
    {-1, ""},
};

static void node_composit_init_curve_vec(bNodeTree *UNUSED(ntree), bNode *node)
{
  node->storage = BKE_curvemapping_add(3, -1.0f, -1.0f, 1.0f, 1.0f);
}

void register_node_type_cmp_curve_vec(void)
{
  static bNodeType ntype;

  cmp_node_type_base(&ntype, CMP_NODE_CURVE_VEC, "Vector Curves", NODE_CLASS_OP_VECTOR, 0);
  node_type_socket_templates(&ntype, cmp_node_curve_vec_in, cmp_node_curve_vec_out);
  node_type_size(&ntype, 200, 140, 320);
  node_type_init(&ntype, node_composit_init_curve_vec);
  node_type_storage(&ntype, "CurveMapping", node_free_curves, node_copy_curves);

  nodeRegisterType(&ntype);
}

/* **************** CURVE RGB  ******************** */

namespace blender::nodes {

static void cmp_node_rgbcurves_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Float>("Fac").default_value(1.0f).min(-1.0f).max(1.0f).subtype(PROP_FACTOR);
  b.add_input<decl::Color>("Image").default_value({1.0f, 1.0f, 1.0f, 1.0f});
  b.add_input<decl::Color>("Black Level").default_value({0.0f, 0.0f, 0.0f, 1.0f});
  b.add_input<decl::Color>("White Level").default_value({1.0f, 1.0f, 1.0f, 1.0f});
  b.add_output<decl::Color>("Image");
}

}  // namespace blender::nodes

static void node_composit_init_curve_rgb(bNodeTree *UNUSED(ntree), bNode *node)
{
  node->storage = BKE_curvemapping_add(4, 0.0f, 0.0f, 1.0f, 1.0f);
}

static int node_composite_gpu_curve_rgb(GPUMaterial *mat,
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

  float start_slopes[CM_TOT];
  float end_slopes[CM_TOT];
  BKE_curvemapping_compute_slopes(curve_mapping, start_slopes, end_slopes);
  float range_minimums[CM_TOT];
  BKE_curvemapping_get_range_minimums(curve_mapping, range_minimums);
  float range_dividers[CM_TOT];
  BKE_curvemapping_compute_range_dividers(curve_mapping, range_dividers);

  if (curve_mapping->tone == CURVE_TONE_FILMLIKE) {
    return GPU_stack_link(mat,
                          node,
                          "curves_film_like",
                          in,
                          out,
                          band_texture,
                          GPU_constant(&band_layer),
                          GPU_uniform(&range_minimums[3]),
                          GPU_uniform(&range_dividers[3]),
                          GPU_uniform(&start_slopes[3]),
                          GPU_uniform(&end_slopes[3]));
  }

  /* If the RGB curves do nothing, use a function that skips RGB computations. */
  if (BKE_curvemapping_is_map_identity(curve_mapping, 0) &&
      BKE_curvemapping_is_map_identity(curve_mapping, 1) &&
      BKE_curvemapping_is_map_identity(curve_mapping, 2)) {
    return GPU_stack_link(mat,
                          node,
                          "curves_combined_only",
                          in,
                          out,
                          band_texture,
                          GPU_constant(&band_layer),
                          GPU_uniform(&range_minimums[3]),
                          GPU_uniform(&range_dividers[3]),
                          GPU_uniform(&start_slopes[3]),
                          GPU_uniform(&end_slopes[3]));
  }

  return GPU_stack_link(mat,
                        node,
                        "curves_combined_rgb",
                        in,
                        out,
                        band_texture,
                        GPU_constant(&band_layer),
                        GPU_uniform(range_minimums),
                        GPU_uniform(range_dividers),
                        GPU_uniform(start_slopes),
                        GPU_uniform(end_slopes));
}

void register_node_type_cmp_curve_rgb(void)
{
  static bNodeType ntype;

  cmp_node_type_base(&ntype, CMP_NODE_CURVE_RGB, "RGB Curves", NODE_CLASS_OP_COLOR, 0);
  ntype.declare = blender::nodes::cmp_node_rgbcurves_declare;
  node_type_size(&ntype, 200, 140, 320);
  node_type_init(&ntype, node_composit_init_curve_rgb);
  node_type_storage(&ntype, "CurveMapping", node_free_curves, node_copy_curves);
  node_type_gpu(&ntype, node_composite_gpu_curve_rgb);

  nodeRegisterType(&ntype);
}
