/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2006 Blender Foundation. All rights reserved. */

/** \file
 * \ingroup cmpnodes
 */

#include "UI_interface.h"
#include "UI_resources.h"

#include "GPU_shader.h"
#include "GPU_texture.h"

#include "COM_node_operation.hh"
#include "COM_utilities.hh"

#include "node_composite_util.hh"

/* **************** FILTER  ******************** */

namespace blender::nodes::node_composite_filter_cc {

static void cmp_node_filter_declare(NodeDeclarationBuilder &b)
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

static void node_composit_buts_filter(uiLayout *layout, bContext *UNUSED(C), PointerRNA *ptr)
{
  uiItemR(layout, ptr, "filter_type", UI_ITEM_R_SPLIT_EMPTY_NAME, "", ICON_NONE);
}

using namespace blender::realtime_compositor;

class FilterOperation : public NodeOperation {
 public:
  using NodeOperation::NodeOperation;

  void execute() override
  {
    GPUShader *shader = shader_pool().acquire("compositor_filter");
    GPU_shader_bind(shader);

    float kernel[3][3];
    get_filter_kernel(kernel);
    GPU_shader_uniform_mat3_as_mat4(shader, "kernel", kernel);

    const Result &input_image = get_input("Image");
    input_image.bind_as_texture(shader, "input_image");

    const Result &factor = get_input("Fac");
    factor.bind_as_texture(shader, "factor");

    const Domain domain = compute_domain();

    Result &output_image = get_result("Image");
    output_image.allocate_texture(domain);
    output_image.bind_as_image(shader, "output_image");

    compute_dispatch_global(shader, domain.size);

    input_image.unbind_as_texture();
    factor.unbind_as_texture();
    output_image.unbind_as_image();
    GPU_shader_unbind();
  }

  int get_filter_method()
  {
    return bnode().custom1;
  }

  void get_filter_kernel(float kernel[3][3])
  {
    switch (get_filter_method()) {
      case CMP_FILT_SOFT:
        kernel[0][0] = 1.0f / 16.0f;
        kernel[0][1] = 2.0f / 16.0f;
        kernel[0][2] = 1.0f / 16.0f;
        kernel[1][0] = 2.0f / 16.0f;
        kernel[1][1] = 4.0f / 16.0f;
        kernel[1][2] = 2.0f / 16.0f;
        kernel[2][0] = 1.0f / 16.0f;
        kernel[2][1] = 2.0f / 16.0f;
        kernel[2][2] = 1.0f / 16.0f;
        break;
      case CMP_FILT_SHARP_BOX:
        kernel[0][0] = -1.0f;
        kernel[0][1] = -1.0f;
        kernel[0][2] = -1.0f;
        kernel[1][0] = -1.0f;
        kernel[1][1] = 9.0f;
        kernel[1][2] = -1.0f;
        kernel[2][0] = -1.0f;
        kernel[2][1] = -1.0f;
        kernel[2][2] = -1.0f;
        break;
      case CMP_FILT_LAPLACE:
        kernel[0][0] = -1.0f / 8.0f;
        kernel[0][1] = -1.0f / 8.0f;
        kernel[0][2] = -1.0f / 8.0f;
        kernel[1][0] = -1.0f / 8.0f;
        kernel[1][1] = 1.0f;
        kernel[1][2] = -1.0f / 8.0f;
        kernel[2][0] = -1.0f / 8.0f;
        kernel[2][1] = -1.0f / 8.0f;
        kernel[2][2] = -1.0f / 8.0f;
        break;
      case CMP_FILT_SOBEL:
        kernel[0][0] = 1.0f;
        kernel[0][1] = 0.0f;
        kernel[0][2] = -1.0f;
        kernel[1][0] = 2.0f;
        kernel[1][1] = 0.0f;
        kernel[1][2] = -2.0f;
        kernel[2][0] = 1.0f;
        kernel[2][1] = 0.0f;
        kernel[2][2] = -1.0f;
        break;
      case CMP_FILT_PREWITT:
        kernel[0][0] = 1.0f;
        kernel[0][1] = 0.0f;
        kernel[0][2] = -1.0f;
        kernel[1][0] = 1.0f;
        kernel[1][1] = 0.0f;
        kernel[1][2] = -1.0f;
        kernel[2][0] = 1.0f;
        kernel[2][1] = 0.0f;
        kernel[2][2] = -1.0f;
        break;
      case CMP_FILT_KIRSCH:
        kernel[0][0] = 5.0f;
        kernel[0][1] = -3.0f;
        kernel[0][2] = -2.0f;
        kernel[1][0] = 5.0f;
        kernel[1][1] = -3.0f;
        kernel[1][2] = -2.0f;
        kernel[2][0] = 5.0f;
        kernel[2][1] = -3.0f;
        kernel[2][2] = -2.0f;
        break;
      case CMP_FILT_SHADOW:
        kernel[0][0] = 1.0f;
        kernel[0][1] = 0.0f;
        kernel[0][2] = -1.0f;
        kernel[1][0] = 2.0f;
        kernel[1][1] = 1.0f;
        kernel[1][2] = -2.0f;
        kernel[2][0] = 1.0f;
        kernel[2][1] = 0.0f;
        kernel[2][2] = -1.0f;
        break;
      case CMP_FILT_SHARP_DIAMOND:
        kernel[0][0] = 0.0f;
        kernel[0][1] = -1.0f;
        kernel[0][2] = 0.0f;
        kernel[1][0] = -1.0f;
        kernel[1][1] = 5.0f;
        kernel[1][2] = -1.0f;
        kernel[2][0] = 0.0f;
        kernel[2][1] = -1.0f;
        kernel[2][2] = 0.0f;
        break;
      default:
        kernel[0][0] = 0.0f;
        kernel[0][1] = 0.0f;
        kernel[0][2] = 0.0f;
        kernel[1][0] = 0.0f;
        kernel[1][1] = 1.0f;
        kernel[1][2] = 0.0f;
        kernel[2][0] = 0.0f;
        kernel[2][1] = 0.0f;
        kernel[2][2] = 0.0f;
        break;
    }
  }
};

static NodeOperation *get_compositor_operation(Context &context, DNode node)
{
  return new FilterOperation(context, node);
}

}  // namespace blender::nodes::node_composite_filter_cc

void register_node_type_cmp_filter()
{
  namespace file_ns = blender::nodes::node_composite_filter_cc;

  static bNodeType ntype;

  cmp_node_type_base(&ntype, CMP_NODE_FILTER, "Filter", NODE_CLASS_OP_FILTER);
  ntype.declare = file_ns::cmp_node_filter_declare;
  ntype.draw_buttons = file_ns::node_composit_buts_filter;
  ntype.labelfunc = node_filter_label;
  ntype.get_compositor_operation = file_ns::get_compositor_operation;
  ntype.flag |= NODE_PREVIEW;

  nodeRegisterType(&ntype);
}
