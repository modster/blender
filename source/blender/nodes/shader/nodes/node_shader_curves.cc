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
 * The Original Code is Copyright (C) 2005 Blender Foundation.
 * All rights reserved.
 */

/** \file
 * \ingroup shdnodes
 */

#include "node_shader_util.h"

namespace blender::nodes {

static void sh_node_curve_vec_declare(NodeDeclarationBuilder &b)
{
  b.is_function_node();
  b.add_input<decl::Float>("Fac").min(0.0f).max(1.0f).default_value(1.0f).subtype(PROP_FACTOR);
  b.add_input<decl::Vector>("Vector").min(-1.0f).max(1.0f);
  b.add_output<decl::Vector>("Vector");
};

}  // namespace blender::nodes

static void node_shader_exec_curve_vec(void *UNUSED(data),
                                       int UNUSED(thread),
                                       bNode *node,
                                       bNodeExecData *UNUSED(execdata),
                                       bNodeStack **in,
                                       bNodeStack **out)
{
  float vec[3];

  /* stack order input:  vec */
  /* stack order output: vec */
  nodestack_get_vec(vec, SOCK_VECTOR, in[1]);
  BKE_curvemapping_evaluate3F((CurveMapping *)node->storage, out[0]->vec, vec);
}

static void node_shader_init_curve_vec(bNodeTree *UNUSED(ntree), bNode *node)
{
  node->storage = BKE_curvemapping_add(3, -1.0f, -1.0f, 1.0f, 1.0f);
}

static int gpu_shader_curve_vec(GPUMaterial *mat,
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

  return GPU_stack_link(mat,
                        node,
                        "curves_vector",
                        in,
                        out,
                        band_texture,
                        GPU_constant(&band_layer),
                        GPU_uniform(range_minimums),
                        GPU_uniform(range_dividers),
                        GPU_uniform(start_slopes),
                        GPU_uniform(end_slopes));
}

class CurveVecFunction : public blender::fn::MultiFunction {
 private:
  const CurveMapping &cumap_;

 public:
  CurveVecFunction(const CurveMapping &cumap) : cumap_(cumap)
  {
    static blender::fn::MFSignature signature = create_signature();
    this->set_signature(&signature);
  }

  static blender::fn::MFSignature create_signature()
  {
    blender::fn::MFSignatureBuilder signature{"Curve Vec"};
    signature.single_input<float>("Fac");
    signature.single_input<blender::float3>("Vector");
    signature.single_output<blender::float3>("Vector");
    return signature.build();
  }

  void call(blender::IndexMask mask,
            blender::fn::MFParams params,
            blender::fn::MFContext UNUSED(context)) const override
  {
    const blender::VArray<float> &fac = params.readonly_single_input<float>(0, "Fac");
    const blender::VArray<blender::float3> &vec_in = params.readonly_single_input<blender::float3>(
        1, "Vector");
    blender::MutableSpan<blender::float3> vec_out =
        params.uninitialized_single_output<blender::float3>(2, "Vector");

    for (int64_t i : mask) {
      BKE_curvemapping_evaluate3F(&cumap_, vec_out[i], vec_in[i]);
      if (fac[i] != 1.0f) {
        interp_v3_v3v3(vec_out[i], vec_in[i], vec_out[i], fac[i]);
      }
    }
  }
};

static void sh_node_curve_vec_build_multi_function(
    blender::nodes::NodeMultiFunctionBuilder &builder)
{
  bNode &bnode = builder.node();
  CurveMapping *cumap = (CurveMapping *)bnode.storage;
  BKE_curvemapping_init(cumap);
  builder.construct_and_set_matching_fn<CurveVecFunction>(*cumap);
}

void register_node_type_sh_curve_vec(void)
{
  static bNodeType ntype;

  sh_fn_node_type_base(&ntype, SH_NODE_CURVE_VEC, "Vector Curves", NODE_CLASS_OP_VECTOR, 0);
  ntype.declare = blender::nodes::sh_node_curve_vec_declare;
  node_type_init(&ntype, node_shader_init_curve_vec);
  node_type_size_preset(&ntype, NODE_SIZE_LARGE);
  node_type_storage(&ntype, "CurveMapping", node_free_curves, node_copy_curves);
  node_type_exec(&ntype, node_initexec_curves, nullptr, node_shader_exec_curve_vec);
  node_type_gpu(&ntype, gpu_shader_curve_vec);
  ntype.build_multi_function = sh_node_curve_vec_build_multi_function;

  nodeRegisterType(&ntype);
}

/* **************** CURVE RGB  ******************** */

namespace blender::nodes {

static void sh_node_curve_rgb_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Float>("Fac").min(0.0f).max(1.0f).default_value(1.0f).subtype(PROP_FACTOR);
  b.add_input<decl::Color>("Color").default_value({1.0f, 1.0f, 1.0f, 1.0f});
  b.add_output<decl::Color>("Color");
};

}  // namespace blender::nodes

static void node_shader_exec_curve_rgb(void *UNUSED(data),
                                       int UNUSED(thread),
                                       bNode *node,
                                       bNodeExecData *UNUSED(execdata),
                                       bNodeStack **in,
                                       bNodeStack **out)
{
  float vec[3];
  float fac;

  /* stack order input:  vec */
  /* stack order output: vec */
  nodestack_get_vec(&fac, SOCK_FLOAT, in[0]);
  nodestack_get_vec(vec, SOCK_VECTOR, in[1]);
  BKE_curvemapping_evaluateRGBF((CurveMapping *)node->storage, out[0]->vec, vec);
  if (fac != 1.0f) {
    interp_v3_v3v3(out[0]->vec, vec, out[0]->vec, fac);
  }
}

static void node_shader_init_curve_rgb(bNodeTree *UNUSED(ntree), bNode *node)
{
  node->storage = BKE_curvemapping_add(4, 0.0f, 0.0f, 1.0f, 1.0f);
}

static int gpu_shader_curve_rgb(GPUMaterial *mat,
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

  /* Shader nodes don't do white balancing. */
  float black_level[4] = {0.0f, 0.0f, 0.0f, 1.0f};
  float white_level[4] = {1.0f, 1.0f, 1.0f, 1.0f};

  /* If the RGB curves do nothing, use a function that skips RGB computations. */
  if (BKE_curvemapping_is_map_identity(curve_mapping, 0) &&
      BKE_curvemapping_is_map_identity(curve_mapping, 1) &&
      BKE_curvemapping_is_map_identity(curve_mapping, 2)) {
    return GPU_stack_link(mat,
                          node,
                          "curves_combined_only",
                          in,
                          out,
                          GPU_constant(black_level),
                          GPU_constant(white_level),
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
                        GPU_constant(black_level),
                        GPU_constant(white_level),
                        band_texture,
                        GPU_constant(&band_layer),
                        GPU_uniform(range_minimums),
                        GPU_uniform(range_dividers),
                        GPU_uniform(start_slopes),
                        GPU_uniform(end_slopes));
}

class CurveRGBFunction : public blender::fn::MultiFunction {
 private:
  const CurveMapping &cumap_;

 public:
  CurveRGBFunction(const CurveMapping &cumap) : cumap_(cumap)
  {
    static blender::fn::MFSignature signature = create_signature();
    this->set_signature(&signature);
  }

  static blender::fn::MFSignature create_signature()
  {
    blender::fn::MFSignatureBuilder signature{"Curve RGB"};
    signature.single_input<float>("Fac");
    signature.single_input<blender::ColorGeometry4f>("Color");
    signature.single_output<blender::ColorGeometry4f>("Color");
    return signature.build();
  }

  void call(blender::IndexMask mask,
            blender::fn::MFParams params,
            blender::fn::MFContext UNUSED(context)) const override
  {
    const blender::VArray<float> &fac = params.readonly_single_input<float>(0, "Fac");
    const blender::VArray<blender::ColorGeometry4f> &col_in =
        params.readonly_single_input<blender::ColorGeometry4f>(1, "Color");
    blender::MutableSpan<blender::ColorGeometry4f> col_out =
        params.uninitialized_single_output<blender::ColorGeometry4f>(2, "Color");

    for (int64_t i : mask) {
      BKE_curvemapping_evaluateRGBF(&cumap_, col_out[i], col_in[i]);
      if (fac[i] != 1.0f) {
        interp_v3_v3v3(col_out[i], col_in[i], col_out[i], fac[i]);
      }
    }
  }
};

static void sh_node_curve_rgb_build_multi_function(
    blender::nodes::NodeMultiFunctionBuilder &builder)
{
  bNode &bnode = builder.node();
  CurveMapping *cumap = (CurveMapping *)bnode.storage;
  BKE_curvemapping_init(cumap);
  builder.construct_and_set_matching_fn<CurveRGBFunction>(*cumap);
}

void register_node_type_sh_curve_rgb(void)
{
  static bNodeType ntype;

  sh_fn_node_type_base(&ntype, SH_NODE_CURVE_RGB, "RGB Curves", NODE_CLASS_OP_COLOR, 0);
  ntype.declare = blender::nodes::sh_node_curve_rgb_declare;
  node_type_init(&ntype, node_shader_init_curve_rgb);
  node_type_size_preset(&ntype, NODE_SIZE_LARGE);
  node_type_storage(&ntype, "CurveMapping", node_free_curves, node_copy_curves);
  node_type_exec(&ntype, node_initexec_curves, nullptr, node_shader_exec_curve_rgb);
  node_type_gpu(&ntype, gpu_shader_curve_rgb);
  ntype.build_multi_function = sh_node_curve_rgb_build_multi_function;

  nodeRegisterType(&ntype);
}
