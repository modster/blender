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

#include "UI_interface.h"
#include "UI_resources.h"

#include "GPU_compute.h"
#include "GPU_shader.h"
#include "GPU_texture.h"

#include "VPC_compositor_execute.hh"

#include "node_composite_util.hh"

/* **************** Flip  ******************** */

namespace blender::nodes::node_composite_flip_cc {

static void cmp_node_flip_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Color>(N_("Image"))
      .default_value({1.0f, 1.0f, 1.0f, 1.0f})
      .compositor_domain_priority(0);
  b.add_output<decl::Color>(N_("Image"));
}

static void node_composit_buts_flip(uiLayout *layout, bContext *UNUSED(C), PointerRNA *ptr)
{
  uiItemR(layout, ptr, "axis", UI_ITEM_R_SPLIT_EMPTY_NAME, "", ICON_NONE);
}

using namespace blender::viewport_compositor;

class FlipOperation : public NodeOperation {
 public:
  using NodeOperation::NodeOperation;

  void execute() override
  {
    Result &input = get_input("Image");
    Result &result = get_result("Image");

    /* Can't flip a single value, pass it through to the output. */
    if (input.is_single_value()) {
      input.pass_through(result);
    }

    GPUShader *shader = get_flip_shader();
    GPU_shader_bind(shader);

    input.bind_as_texture(shader, "input_image");

    result.allocate_texture(compute_domain());
    result.bind_as_image(shader, "output_image");

    const int2 size = compute_domain().size;
    GPU_compute_dispatch(shader, size.x / 16 + 1, size.y / 16 + 1, 1);

    input.unbind_as_texture();
    result.unbind_as_image();
    GPU_shader_unbind();
    GPU_shader_free(shader);
  }

  GPUShader *get_flip_shader()
  {
    switch (get_flip_mode()) {
      case 0:
        return GPU_shader_create_from_info_name("compositor_flip_x");
      case 1:
        return GPU_shader_create_from_info_name("compositor_flip_y");
      case 2:
        return GPU_shader_create_from_info_name("compositor_flip_x_and_y");
    }

    BLI_assert_unreachable();
    return nullptr;
  }

  /* 0 -> Flip along x.
   * 1 -> Flip along y.
   * 2 -> Flip along both x and y. */
  int get_flip_mode()
  {
    return node().custom1;
  }
};

static NodeOperation *get_compositor_operation(Context &context, DNode node)
{
  return new FlipOperation(context, node);
}

}  // namespace blender::nodes::node_composite_flip_cc

void register_node_type_cmp_flip()
{
  namespace file_ns = blender::nodes::node_composite_flip_cc;

  static bNodeType ntype;

  cmp_node_type_base(&ntype, CMP_NODE_FLIP, "Flip", NODE_CLASS_DISTORT);
  ntype.declare = file_ns::cmp_node_flip_declare;
  ntype.draw_buttons = file_ns::node_composit_buts_flip;
  ntype.get_compositor_operation = file_ns::get_compositor_operation;

  nodeRegisterType(&ntype);
}
