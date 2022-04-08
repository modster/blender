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

#include "BLI_math_vec_types.hh"
#include "BLI_transformation_2d.hh"

#include "UI_interface.h"
#include "UI_resources.h"

#include "VPC_compositor_execute.hh"

#include "node_composite_util.hh"

/* **************** Translate ******************** */

namespace blender::nodes::node_composite_translate_cc {

static void cmp_node_translate_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Color>(N_("Image"))
      .default_value({1.0f, 1.0f, 1.0f, 1.0f})
      .compositor_domain_priority(0);
  b.add_input<decl::Float>(N_("X"))
      .default_value(0.0f)
      .min(-10000.0f)
      .max(10000.0f)
      .compositor_expects_single_value();
  b.add_input<decl::Float>(N_("Y"))
      .default_value(0.0f)
      .min(-10000.0f)
      .max(10000.0f)
      .compositor_expects_single_value();
  b.add_output<decl::Color>(N_("Image"));
}

static void node_composit_init_translate(bNodeTree *UNUSED(ntree), bNode *node)
{
  NodeTranslateData *data = MEM_cnew<NodeTranslateData>(__func__);
  node->storage = data;
}

static void node_composit_buts_translate(uiLayout *layout, bContext *UNUSED(C), PointerRNA *ptr)
{
  uiItemR(layout, ptr, "use_relative", UI_ITEM_R_SPLIT_EMPTY_NAME, nullptr, ICON_NONE);
  uiItemR(layout, ptr, "wrap_axis", UI_ITEM_R_SPLIT_EMPTY_NAME, nullptr, ICON_NONE);
}

using namespace blender::viewport_compositor;

class TranslateOperation : public NodeOperation {
 public:
  using NodeOperation::NodeOperation;

  void execute() override
  {
    Result &input = get_input("Image");
    Result &result = get_result("Image");
    input.pass_through(result);

    float x = get_input("X").get_float_value_default(0.0f);
    float y = get_input("Y").get_float_value_default(0.0f);
    if (get_use_relative()) {
      x *= input.domain().size.x;
      y *= input.domain().size.y;
    }

    const float2 translation = float2(x, y);
    const Transformation2D transformation = Transformation2D::from_translation(translation);

    result.transform(transformation);
    result.get_realization_options().repeat_x = get_repeat_x();
    result.get_realization_options().repeat_y = get_repeat_y();
  }

  NodeTranslateData &get_node_translate()
  {
    return *static_cast<NodeTranslateData *>(node().storage);
  }

  bool get_use_relative()
  {
    return get_node_translate().relative;
  }

  bool get_repeat_x()
  {
    return ELEM(get_node_translate().wrap_axis, CMP_NODE_WRAP_X, CMP_NODE_WRAP_XY);
  }

  bool get_repeat_y()
  {
    return ELEM(get_node_translate().wrap_axis, CMP_NODE_WRAP_Y, CMP_NODE_WRAP_XY);
  }
};

static NodeOperation *get_compositor_operation(Context &context, DNode node)
{
  return new TranslateOperation(context, node);
}

}  // namespace blender::nodes::node_composite_translate_cc

void register_node_type_cmp_translate()
{
  namespace file_ns = blender::nodes::node_composite_translate_cc;

  static bNodeType ntype;

  cmp_node_type_base(&ntype, CMP_NODE_TRANSLATE, "Translate", NODE_CLASS_DISTORT);
  ntype.declare = file_ns::cmp_node_translate_declare;
  ntype.draw_buttons = file_ns::node_composit_buts_translate;
  node_type_init(&ntype, file_ns::node_composit_init_translate);
  node_type_storage(
      &ntype, "NodeTranslateData", node_free_standard_storage, node_copy_standard_storage);
  ntype.get_compositor_operation = file_ns::get_compositor_operation;

  nodeRegisterType(&ntype);
}
