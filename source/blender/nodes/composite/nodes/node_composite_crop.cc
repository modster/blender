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

#include "BLI_math_base.h"
#include "BLI_math_vec_types.hh"

#include "DNA_node_types.h"

#include "RNA_access.h"

#include "UI_interface.h"
#include "UI_resources.h"

#include "GPU_compute.h"
#include "GPU_shader.h"
#include "GPU_texture.h"

#include "NOD_compositor_execute.hh"

#include "node_composite_util.hh"

/* **************** Crop  ******************** */

namespace blender::nodes::node_composite_crop_cc {

static void cmp_node_crop_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Color>(N_("Image"))
      .default_value({1.0f, 1.0f, 1.0f, 1.0f})
      .compositor_domain_priority(0);
  b.add_output<decl::Color>(N_("Image"));
}

static void node_composit_init_crop(bNodeTree *UNUSED(ntree), bNode *node)
{
  NodeTwoXYs *nxy = MEM_cnew<NodeTwoXYs>(__func__);
  node->storage = nxy;
  nxy->x1 = 0;
  nxy->x2 = 0;
  nxy->y1 = 0;
  nxy->y2 = 0;
}

static void node_composit_buts_crop(uiLayout *layout, bContext *UNUSED(C), PointerRNA *ptr)
{
  uiLayout *col;

  uiItemR(layout, ptr, "use_crop_size", UI_ITEM_R_SPLIT_EMPTY_NAME, nullptr, ICON_NONE);
  uiItemR(layout, ptr, "relative", UI_ITEM_R_SPLIT_EMPTY_NAME, nullptr, ICON_NONE);

  col = uiLayoutColumn(layout, true);
  if (RNA_boolean_get(ptr, "relative")) {
    uiItemR(col, ptr, "rel_min_x", UI_ITEM_R_SPLIT_EMPTY_NAME, IFACE_("Left"), ICON_NONE);
    uiItemR(col, ptr, "rel_max_x", UI_ITEM_R_SPLIT_EMPTY_NAME, IFACE_("Right"), ICON_NONE);
    uiItemR(col, ptr, "rel_min_y", UI_ITEM_R_SPLIT_EMPTY_NAME, IFACE_("Up"), ICON_NONE);
    uiItemR(col, ptr, "rel_max_y", UI_ITEM_R_SPLIT_EMPTY_NAME, IFACE_("Down"), ICON_NONE);
  }
  else {
    uiItemR(col, ptr, "min_x", UI_ITEM_R_SPLIT_EMPTY_NAME, IFACE_("Left"), ICON_NONE);
    uiItemR(col, ptr, "max_x", UI_ITEM_R_SPLIT_EMPTY_NAME, IFACE_("Right"), ICON_NONE);
    uiItemR(col, ptr, "min_y", UI_ITEM_R_SPLIT_EMPTY_NAME, IFACE_("Up"), ICON_NONE);
    uiItemR(col, ptr, "max_y", UI_ITEM_R_SPLIT_EMPTY_NAME, IFACE_("Down"), ICON_NONE);
  }
}

using namespace blender::viewport_compositor;

class CropOperation : public NodeOperation {
 public:
  using NodeOperation::NodeOperation;

  void execute() override
  {
    /* The operation does nothing, so just pass the input through. */
    if (is_identity()) {
      get_input("Image").pass_through(get_result("Image"));
    }

    if (get_is_image_crop()) {
      execute_image_crop();
    }
    else {
      execute_alpha_crop();
    }
  }

  /* Crop by replacing areas outside of the cropping bounds with zero alpha. The output have the
   * same domain as the input image. */
  void execute_alpha_crop()
  {
    GPUShader *shader = GPU_shader_create_from_info_name("compositor_alpha_crop");
    GPU_shader_bind(shader);

    int2 lower_bound, upper_bound;
    compute_cropping_bounds(lower_bound, upper_bound);
    GPU_shader_uniform_2iv(shader, "lower_bound", lower_bound);
    GPU_shader_uniform_2iv(shader, "upper_bound", upper_bound);

    const Result &input_image = get_input("Image");
    input_image.bind_as_texture(shader, "input_image");

    Result &output_image = get_result("Image");
    output_image.allocate_texture(compute_domain());
    output_image.bind_as_image(shader, "output_image");

    const int2 size = compute_domain().size;
    GPU_compute_dispatch(shader, size.x / 16 + 1, size.y / 16 + 1, 1);

    input_image.unbind_as_texture();
    output_image.unbind_as_image();
    GPU_shader_unbind();
    GPU_shader_free(shader);
  }

  /* Crop the image into a new size that matches the cropping bounds. */
  void execute_image_crop()
  {
    int2 lower_bound, upper_bound;
    compute_cropping_bounds(lower_bound, upper_bound);

    /* The image is cropped into nothing, so just return a single zero value. */
    if (lower_bound.x == upper_bound.x || lower_bound.y == upper_bound.y) {
      Result &result = get_result("Image");
      result.allocate_single_value();
      result.set_color_value(float4(0.0f));
      return;
    }

    GPUShader *shader = GPU_shader_create_from_info_name("compositor_image_crop");
    GPU_shader_bind(shader);

    GPU_shader_uniform_2iv(shader, "lower_bound", lower_bound);

    const Result &input_image = get_input("Image");
    input_image.bind_as_texture(shader, "input_image");

    const int2 size = upper_bound - lower_bound;

    Result &output_image = get_result("Image");
    output_image.allocate_texture(Domain(size, compute_domain().transformation));
    output_image.bind_as_image(shader, "output_image");

    GPU_compute_dispatch(shader, size.x / 16 + 1, size.y / 16 + 1, 1);

    input_image.unbind_as_texture();
    output_image.unbind_as_image();
    GPU_shader_unbind();
    GPU_shader_free(shader);
  }

  /* If true, the image should actually be cropped into a new size. Otherwise, if false, the region
   * outside of the cropping bounds will be set to a zero alpha value. */
  bool get_is_image_crop()
  {
    return node().custom1;
  }

  bool get_is_relative()
  {
    return node().custom2;
  }

  NodeTwoXYs &get_node_two_xys()
  {
    return *static_cast<NodeTwoXYs *>(node().storage);
  }

  /* Returns true if the operation does nothing and the input can be passed through. */
  bool is_identity()
  {
    const Result &input = get_input("Image");
    /* Single value inputs can't be cropped and are returned as is. */
    if (input.is_single_value()) {
      return true;
    }

    int2 lower_bound, upper_bound;
    compute_cropping_bounds(lower_bound, upper_bound);
    const int2 input_size = input.domain().size;
    /* The cropping bounds cover the whole image, so no cropping happens. */
    if (lower_bound == int2(0) && upper_bound == input_size) {
      return true;
    }

    return false;
  }

  void compute_cropping_bounds(int2 &lower_bound, int2 &upper_bound)
  {
    const NodeTwoXYs &node_two_xys = get_node_two_xys();
    const int2 input_size = get_input("Image").domain().size;

    if (get_is_relative()) {
      /* The cropping bounds are relative to the image size. The factors are in the [0, 1] range,
       * so it is guaranteed that they won't go over the input image size. */
      lower_bound.x = input_size.x * node_two_xys.fac_x1;
      lower_bound.y = input_size.y * node_two_xys.fac_y2;
      upper_bound.x = input_size.x * node_two_xys.fac_x2;
      upper_bound.y = input_size.y * node_two_xys.fac_y1;
    }
    else {
      /* Make sure the bounds don't go over the input image size. */
      lower_bound.x = min_ii(node_two_xys.x1, input_size.x);
      lower_bound.y = min_ii(node_two_xys.y2, input_size.y);
      upper_bound.x = min_ii(node_two_xys.x2, input_size.x);
      upper_bound.y = min_ii(node_two_xys.y1, input_size.y);
    }

    /* Make sure upper bound is actually higher than the lower bound. */
    lower_bound.x = min_ii(lower_bound.x, upper_bound.x);
    lower_bound.y = min_ii(lower_bound.y, upper_bound.y);
    upper_bound.x = max_ii(lower_bound.x, upper_bound.x);
    upper_bound.y = max_ii(lower_bound.y, upper_bound.y);
  }
};

static NodeOperation *get_compositor_operation(Context &context, DNode node)
{
  return new CropOperation(context, node);
}

}  // namespace blender::nodes::node_composite_crop_cc

void register_node_type_cmp_crop()
{
  namespace file_ns = blender::nodes::node_composite_crop_cc;

  static bNodeType ntype;

  cmp_node_type_base(&ntype, CMP_NODE_CROP, "Crop", NODE_CLASS_DISTORT);
  ntype.declare = file_ns::cmp_node_crop_declare;
  ntype.draw_buttons = file_ns::node_composit_buts_crop;
  node_type_init(&ntype, file_ns::node_composit_init_crop);
  node_type_storage(&ntype, "NodeTwoXYs", node_free_standard_storage, node_copy_standard_storage);
  ntype.get_compositor_operation = file_ns::get_compositor_operation;

  nodeRegisterType(&ntype);
}
