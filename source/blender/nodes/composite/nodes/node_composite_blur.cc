/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2006 Blender Foundation. All rights reserved. */

/** \file
 * \ingroup cmpnodes
 */

#include "BLI_assert.h"
#include "BLI_math_base.hh"
#include "BLI_math_vec_types.hh"
#include "BLI_vector.hh"

#include "DNA_scene_types.h"

#include "RE_pipeline.h"

#include "RNA_access.h"

#include "UI_interface.h"
#include "UI_resources.h"

#include "GPU_state.h"
#include "GPU_texture.h"

#include "VPC_node_operation.hh"
#include "VPC_utilities.hh"

#include "node_composite_util.hh"

/* **************** BLUR ******************** */

namespace blender::nodes::node_composite_blur_cc {

static void cmp_node_blur_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Color>(N_("Image")).default_value({1.0f, 1.0f, 1.0f, 1.0f});
  b.add_input<decl::Float>(N_("Size")).default_value(1.0f).min(0.0f).max(1.0f);
  b.add_output<decl::Color>(N_("Image"));
}

static void node_composit_init_blur(bNodeTree *UNUSED(ntree), bNode *node)
{
  NodeBlurData *data = MEM_cnew<NodeBlurData>(__func__);
  data->filtertype = R_FILTER_GAUSS;
  node->storage = data;
}

static void node_composit_buts_blur(uiLayout *layout, bContext *UNUSED(C), PointerRNA *ptr)
{
  uiLayout *col, *row;

  col = uiLayoutColumn(layout, false);
  const int filter = RNA_enum_get(ptr, "filter_type");
  const int reference = RNA_boolean_get(ptr, "use_variable_size");

  uiItemR(col, ptr, "filter_type", UI_ITEM_R_SPLIT_EMPTY_NAME, "", ICON_NONE);
  if (filter != R_FILTER_FAST_GAUSS) {
    uiItemR(col, ptr, "use_variable_size", UI_ITEM_R_SPLIT_EMPTY_NAME, nullptr, ICON_NONE);
    if (!reference) {
      uiItemR(col, ptr, "use_bokeh", UI_ITEM_R_SPLIT_EMPTY_NAME, nullptr, ICON_NONE);
    }
    uiItemR(col, ptr, "use_gamma_correction", UI_ITEM_R_SPLIT_EMPTY_NAME, nullptr, ICON_NONE);
  }

  uiItemR(col, ptr, "use_relative", UI_ITEM_R_SPLIT_EMPTY_NAME, nullptr, ICON_NONE);

  if (RNA_boolean_get(ptr, "use_relative")) {
    uiItemL(col, IFACE_("Aspect Correction"), ICON_NONE);
    row = uiLayoutRow(layout, true);
    uiItemR(row,
            ptr,
            "aspect_correction",
            UI_ITEM_R_SPLIT_EMPTY_NAME | UI_ITEM_R_EXPAND,
            nullptr,
            ICON_NONE);

    col = uiLayoutColumn(layout, true);
    uiItemR(col, ptr, "factor_x", UI_ITEM_R_SPLIT_EMPTY_NAME, IFACE_("X"), ICON_NONE);
    uiItemR(col, ptr, "factor_y", UI_ITEM_R_SPLIT_EMPTY_NAME, IFACE_("Y"), ICON_NONE);
  }
  else {
    col = uiLayoutColumn(layout, true);
    uiItemR(col, ptr, "size_x", UI_ITEM_R_SPLIT_EMPTY_NAME, IFACE_("X"), ICON_NONE);
    uiItemR(col, ptr, "size_y", UI_ITEM_R_SPLIT_EMPTY_NAME, IFACE_("Y"), ICON_NONE);
  }
  uiItemR(col, ptr, "use_extended_bounds", UI_ITEM_R_SPLIT_EMPTY_NAME, nullptr, ICON_NONE);
}

using namespace blender::viewport_compositor;

class BlurWeights {
 public:
  int type_;
  float radius_;
  GPUTexture *texture_ = nullptr;

  ~BlurWeights()
  {
    if (texture_) {
      GPU_texture_free(texture_);
    }
  }

  void bind_as_texture(GPUShader *shader, const char *texture_name)
  {
    const int texture_image_unit = GPU_shader_get_texture_binding(shader, texture_name);
    GPU_texture_bind(texture_, texture_image_unit);
  }

  void unbind_as_texture()
  {
    GPU_texture_unbind(texture_);
  }

  void update(float radius, int type)
  {
    if (texture_ && type == type_ && radius == radius_) {
      return;
    }

    if (texture_) {
      GPU_texture_free(texture_);
    }

    const int size = ceil(radius);
    Vector<float> weights(2 * size + 1);

    float sum = 0.0f;
    const float scale = radius > 0.0f ? 1.0f / radius : 0.0f;
    for (int i = -size; i <= size; i++) {
      const float weight = RE_filter_value(type, i * scale);
      sum += weight;
      weights[i + size] = weight;
    }

    for (int i = 0; i < weights.size(); i++) {
      weights[i] /= sum;
    }

    texture_ = GPU_texture_create_1d("Weights", weights.size(), 1, GPU_R32F, weights.data());

    type_ = type;
    radius_ = radius;
  }
};

class BlurOperation : public NodeOperation {
 private:
  /* Cached blur weights for the horizontal pass. */
  BlurWeights horizontal_weights_;
  /* Cached blur weights for the vertical pass. */
  BlurWeights vertical_weights_;

 public:
  using NodeOperation::NodeOperation;

  void execute() override
  {
    if (is_identity()) {
      get_input("Image").pass_through(get_result("Image"));
      return;
    }

    GPUTexture *horizontal_pass_result = execute_horizontal_pass();
    execute_vertical_pass(horizontal_pass_result);
  }

  /* Blur the input image horizontally using the horizontal weights, write the output into an
   * intermediate texture, and return it. */
  GPUTexture *execute_horizontal_pass()
  {
    GPUShader *shader = shader_pool().acquire("compositor_blur_horizontal");
    GPU_shader_bind(shader);

    const Result &input_image = get_input("Image");
    input_image.bind_as_texture(shader, "input_image");

    horizontal_weights_.update(compute_blur_radius().x, get_blur_data().filtertype);
    horizontal_weights_.bind_as_texture(shader, "weights");

    const Domain domain = compute_domain();

    GPUTexture *horizontal_pass_result = texture_pool().acquire_color(domain.size);
    const int image_unit = GPU_shader_get_texture_binding(shader, "output_image");
    GPU_texture_image_bind(horizontal_pass_result, image_unit);

    compute_dispatch_global(shader, domain.size);

    GPU_shader_unbind();
    input_image.unbind_as_texture();
    horizontal_weights_.unbind_as_texture();
    GPU_texture_image_unbind(horizontal_pass_result);

    return horizontal_pass_result;
  }

  /* Blur the intermediate texture returned by the horizontal pass using the vertical weights and
   * write the output into the result. */
  void execute_vertical_pass(GPUTexture *horizontal_pass_result)
  {
    GPUShader *shader = shader_pool().acquire("compositor_blur_vertical");
    GPU_shader_bind(shader);

    GPU_memory_barrier(GPU_BARRIER_TEXTURE_FETCH);
    const int texture_image_unit = GPU_shader_get_texture_binding(shader, "input_image");
    GPU_texture_bind(horizontal_pass_result, texture_image_unit);

    vertical_weights_.update(compute_blur_radius().y, get_blur_data().filtertype);
    vertical_weights_.bind_as_texture(shader, "weights");

    const Domain domain = compute_domain();

    Result &output_image = get_result("Image");
    output_image.allocate_texture(domain);
    output_image.bind_as_image(shader, "output_image");

    compute_dispatch_global(shader, domain.size);

    GPU_shader_unbind();
    output_image.unbind_as_image();
    vertical_weights_.unbind_as_texture();
    GPU_texture_unbind(horizontal_pass_result);
  }

  float2 compute_blur_radius()
  {
    const float size = get_input("Size").get_float_value_default(1.0f);
    if (!get_blur_data().relative) {
      return float2(get_blur_data().sizex, get_blur_data().sizey) * size;
    }

    int2 input_size = get_input("Image").domain().size;
    switch (get_blur_data().aspect) {
      case CMP_NODE_BLUR_ASPECT_Y:
        input_size.y = input_size.x;
        break;
      case CMP_NODE_BLUR_ASPECT_X:
        input_size.x = input_size.y;
        break;
      default:
        BLI_assert(get_blur_data().aspect == CMP_NODE_BLUR_ASPECT_NONE);
        break;
    }
    return float2(input_size) * get_size_factor() * size;
  }

  float2 get_size_factor()
  {
    return float2(get_blur_data().percentx, get_blur_data().percenty) / 100.0f;
  }

  NodeBlurData &get_blur_data()
  {
    return *static_cast<NodeBlurData *>(bnode().storage);
  }

  /* Returns true if the operation does nothing and the input can be passed through. */
  bool is_identity()
  {
    const Result &input = get_input("Image");
    /* Single value inputs can't be blurred and are returned as is. */
    if (input.is_single_value()) {
      return true;
    }

    /* Zero blur radius. The operation does nothing and the input can be passed through. */
    if (compute_blur_radius() == float2(0.0)) {
      return true;
    }

    return false;
  }
};

static NodeOperation *get_compositor_operation(Context &context, DNode node)
{
  return new BlurOperation(context, node);
}

}  // namespace blender::nodes::node_composite_blur_cc

void register_node_type_cmp_blur()
{
  namespace file_ns = blender::nodes::node_composite_blur_cc;

  static bNodeType ntype;

  cmp_node_type_base(&ntype, CMP_NODE_BLUR, "Blur", NODE_CLASS_OP_FILTER);
  ntype.declare = file_ns::cmp_node_blur_declare;
  ntype.draw_buttons = file_ns::node_composit_buts_blur;
  ntype.flag |= NODE_PREVIEW;
  node_type_init(&ntype, file_ns::node_composit_init_blur);
  node_type_storage(
      &ntype, "NodeBlurData", node_free_standard_storage, node_copy_standard_storage);
  ntype.get_compositor_operation = file_ns::get_compositor_operation;

  nodeRegisterType(&ntype);
}
