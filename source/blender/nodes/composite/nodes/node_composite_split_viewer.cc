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

#include "BKE_global.h"
#include "BKE_image.h"

#include "UI_interface.h"
#include "UI_resources.h"

#include "GPU_compute.h"
#include "GPU_shader.h"
#include "GPU_texture.h"

#include "NOD_compositor_execute.hh"

#include "node_composite_util.hh"

/* **************** SPLIT VIEWER ******************** */

namespace blender::nodes::node_composite_split_viewer_cc {

static void cmp_node_split_viewer_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Color>(N_("Image"));
  b.add_input<decl::Color>(N_("Image"), "Image_001");
}

static void node_composit_init_splitviewer(bNodeTree *UNUSED(ntree), bNode *node)
{
  ImageUser *iuser = MEM_cnew<ImageUser>(__func__);
  node->storage = iuser;
  iuser->sfra = 1;
  node->custom1 = 50; /* default 50% split */

  node->id = (ID *)BKE_image_ensure_viewer(G.main, IMA_TYPE_COMPOSITE, "Viewer Node");
}

static void node_composit_buts_splitviewer(uiLayout *layout, bContext *UNUSED(C), PointerRNA *ptr)
{
  uiLayout *row, *col;

  col = uiLayoutColumn(layout, false);
  row = uiLayoutRow(col, false);
  uiItemR(row, ptr, "axis", UI_ITEM_R_SPLIT_EMPTY_NAME | UI_ITEM_R_EXPAND, nullptr, ICON_NONE);
  uiItemR(col, ptr, "factor", UI_ITEM_R_SPLIT_EMPTY_NAME, nullptr, ICON_NONE);
}

using namespace blender::viewport_compositor;

class ViewerOperation : public NodeOperation {
 public:
  using NodeOperation::NodeOperation;

  void execute() override
  {
    GPUShader *shader = get_split_viewer_shader();
    GPU_shader_bind(shader);

    const int2 size = compute_domain().size;

    GPU_shader_uniform_1f(shader, "split_ratio", get_split_ratio());
    GPU_shader_uniform_2iv(shader, "view_size", size);

    const Result &first_image = get_input("Image");
    first_image.bind_as_texture(shader, "first_image");
    const Result &second_image = get_input("Image_001");
    second_image.bind_as_texture(shader, "second_image");

    GPUTexture *viewport_texture = context().get_viewport_texture();
    const int image_unit = GPU_shader_get_texture_binding(shader, "output_image");
    GPU_texture_image_bind(viewport_texture, image_unit);

    GPU_compute_dispatch(shader, size.x / 16 + 1, size.y / 16 + 1, 1);

    first_image.unbind_as_texture();
    second_image.unbind_as_texture();
    GPU_texture_image_unbind(viewport_texture);
    GPU_shader_unbind();
    GPU_shader_free(shader);
  }

  /* The operation domain have the same dimensions of the viewport without any transformations. */
  Domain compute_domain() override
  {
    GPUTexture *viewport_texture = context().get_viewport_texture();
    return Domain(int2(GPU_texture_width(viewport_texture), GPU_texture_height(viewport_texture)));
  }

  GPUShader *get_split_viewer_shader()
  {
    if (get_split_axis() == 0) {
      return GPU_shader_create_from_info_name("compositor_split_viewer_horizontal");
    }

    return GPU_shader_create_from_info_name("compositor_split_viewer_vertical");
  }

  /* 0 -> Split Horizontal.
   * 1 -> Split Vertical. */
  int get_split_axis()
  {
    return node().custom2;
  }

  float get_split_ratio()
  {
    return node().custom1 / 100.0f;
  }
};

static NodeOperation *get_compositor_operation(Context &context, DNode node)
{
  return new ViewerOperation(context, node);
}

}  // namespace blender::nodes::node_composite_split_viewer_cc

void register_node_type_cmp_splitviewer()
{
  namespace file_ns = blender::nodes::node_composite_split_viewer_cc;

  static bNodeType ntype;

  cmp_node_type_base(&ntype, CMP_NODE_SPLITVIEWER, "Split Viewer", NODE_CLASS_OUTPUT);
  ntype.declare = file_ns::cmp_node_split_viewer_declare;
  ntype.draw_buttons = file_ns::node_composit_buts_splitviewer;
  ntype.flag |= NODE_PREVIEW;
  node_type_init(&ntype, file_ns::node_composit_init_splitviewer);
  node_type_storage(&ntype, "ImageUser", node_free_standard_storage, node_copy_standard_storage);
  ntype.get_compositor_operation = file_ns::get_compositor_operation;

  ntype.no_muting = true;

  nodeRegisterType(&ntype);
}
