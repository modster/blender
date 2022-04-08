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

#include "UI_interface.h"
#include "UI_resources.h"

#include "GPU_compute.h"
#include "GPU_shader.h"
#include "GPU_texture.h"

#include "VPC_compositor_execute.hh"

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

using namespace blender::viewport_compositor;

class FilterOperation : public NodeOperation {
 public:
  using NodeOperation::NodeOperation;

  void execute() override
  {
    GPUShader *shader = GPU_shader_create_from_info_name("compositor_filter");
    GPU_shader_bind(shader);

    float kernel[3][3];
    get_filter_kernel(kernel);
    GPU_shader_uniform_mat3(shader, "kernel", kernel);

    const Result &input_image = get_input("Image");
    input_image.bind_as_texture(shader, "input_image");

    const Result &factor = get_input("Fac");
    factor.bind_as_texture(shader, "factor");

    Result &output_image = get_result("Image");
    output_image.allocate_texture(compute_domain());
    output_image.bind_as_image(shader, "output_image");

    const int2 size = compute_domain().size;
    GPU_compute_dispatch(shader, size.x / 16 + 1, size.y / 16 + 1, 1);

    input_image.unbind_as_texture();
    factor.unbind_as_texture();
    output_image.unbind_as_image();
    GPU_shader_unbind();
    GPU_shader_free(shader);
  }

  int get_filter_method()
  {
    return node().custom1;
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
      case CMP_FILT_SHARP:
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
