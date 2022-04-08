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
 * The Original Code is Copyright (C) 2011 Blender Foundation.
 * All rights reserved.
 */

/** \file
 * \ingroup cmpnodes
 */

#include "BLI_math_vec_types.hh"

#include "BKE_context.h"
#include "BKE_lib_id.h"
#include "BKE_movieclip.h"

#include "DNA_defaults.h"

#include "RNA_access.h"

#include "UI_interface.h"
#include "UI_resources.h"

#include "GPU_compute.h"
#include "GPU_shader.h"
#include "GPU_texture.h"

#include "VPC_compositor_execute.hh"

#include "node_composite_util.hh"

namespace blender::nodes::node_composite_movieclip_cc {

static void cmp_node_movieclip_declare(NodeDeclarationBuilder &b)
{
  b.add_output<decl::Color>(N_("Image"));
  b.add_output<decl::Float>(N_("Alpha"));
  b.add_output<decl::Float>(N_("Offset X"));
  b.add_output<decl::Float>(N_("Offset Y"));
  b.add_output<decl::Float>(N_("Scale"));
  b.add_output<decl::Float>(N_("Angle"));
}

static void init(const bContext *C, PointerRNA *ptr)
{
  bNode *node = (bNode *)ptr->data;
  Scene *scene = CTX_data_scene(C);
  MovieClipUser *user = DNA_struct_default_alloc(MovieClipUser);

  node->id = (ID *)scene->clip;
  id_us_plus(node->id);
  node->storage = user;
  user->framenr = 1;
}

static void node_composit_buts_movieclip(uiLayout *layout, bContext *C, PointerRNA *ptr)
{
  uiTemplateID(layout,
               C,
               ptr,
               "clip",
               nullptr,
               "CLIP_OT_open",
               nullptr,
               UI_TEMPLATE_ID_FILTER_ALL,
               false,
               nullptr);
}

static void node_composit_buts_movieclip_ex(uiLayout *layout, bContext *C, PointerRNA *ptr)
{
  bNode *node = (bNode *)ptr->data;
  PointerRNA clipptr;

  uiTemplateID(layout,
               C,
               ptr,
               "clip",
               nullptr,
               "CLIP_OT_open",
               nullptr,
               UI_TEMPLATE_ID_FILTER_ALL,
               false,
               nullptr);

  if (!node->id) {
    return;
  }

  clipptr = RNA_pointer_get(ptr, "clip");

  uiTemplateColorspaceSettings(layout, &clipptr, "colorspace_settings");
}

using namespace blender::viewport_compositor;

class MovieClipOperation : public NodeOperation {
 public:
  using NodeOperation::NodeOperation;

  void execute() override
  {
    GPUTexture *movie_clip_texture = get_movie_clip_texture();

    compute_image(movie_clip_texture);
    compute_alpha(movie_clip_texture);
    compute_stabilization_data(movie_clip_texture);

    free_movie_clip_texture();
  }

  void compute_image(GPUTexture *movie_clip_texture)
  {
    if (!is_output_needed("Image")) {
      return;
    }

    Result &result = get_result("Image");

    /* The movie clip texture is invalid or missing, set an appropriate fallback value. */
    if (!movie_clip_texture) {
      result.allocate_single_value();
      result.set_color_value(float4(float3(0.0), 1.0f));
      return;
    }

    const int width = GPU_texture_width(movie_clip_texture);
    const int height = GPU_texture_height(movie_clip_texture);
    result.allocate_texture(Domain(int2(width, height)));

    /* The movie clip texture already has an appropriate half float format, so just copy. */
    if (GPU_texture_format(movie_clip_texture) == GPU_RGBA16F) {
      GPU_texture_copy(result.texture(), movie_clip_texture);
      return;
    }

    GPUShader *shader = GPU_shader_create_from_info_name("compositor_convert_color_to_half_color");
    GPU_shader_bind(shader);

    const int input_unit = GPU_shader_get_texture_binding(shader, "input_sampler");
    GPU_texture_bind(movie_clip_texture, input_unit);

    result.bind_as_image(shader, "output_image");

    GPU_compute_dispatch(shader, width / 16 + 1, height / 16 + 1, 1);

    GPU_shader_unbind();
    GPU_texture_unbind(movie_clip_texture);
    result.unbind_as_image();
    GPU_shader_free(shader);
  }

  void compute_alpha(GPUTexture *movie_clip_texture)
  {
    if (!is_output_needed("Alpha")) {
      return;
    }

    Result &result = get_result("Alpha");

    /* The movie clip texture is invalid or missing, set an appropriate fallback value. */
    if (!movie_clip_texture) {
      result.allocate_single_value();
      result.set_float_value(1.0f);
      return;
    }

    const int width = GPU_texture_width(movie_clip_texture);
    const int height = GPU_texture_height(movie_clip_texture);
    result.allocate_texture(Domain(int2(width, height)));

    GPUShader *shader = GPU_shader_create_from_info_name("compositor_convert_color_to_alpha");
    GPU_shader_bind(shader);

    const int input_unit = GPU_shader_get_texture_binding(shader, "input_sampler");
    GPU_texture_bind(movie_clip_texture, input_unit);

    result.bind_as_image(shader, "output_image");

    GPU_compute_dispatch(shader, width / 16 + 1, height / 16 + 1, 1);

    GPU_shader_unbind();
    GPU_texture_unbind(movie_clip_texture);
    result.unbind_as_image();
    GPU_shader_free(shader);
  }

  void compute_stabilization_data(GPUTexture *movie_clip_texture)
  {
    /* The movie clip texture is invalid or missing, set appropriate fallback values. */
    if (!movie_clip_texture) {
      if (is_output_needed("Offset X")) {
        Result &result = get_result("Offset X");
        result.allocate_single_value();
        result.set_float_value(0.0f);
      }
      if (is_output_needed("Offset Y")) {
        Result &result = get_result("Offset Y");
        result.allocate_single_value();
        result.set_float_value(0.0f);
      }
      if (is_output_needed("Scale")) {
        Result &result = get_result("Scale");
        result.allocate_single_value();
        result.set_float_value(1.0f);
      }
      if (is_output_needed("Angle")) {
        Result &result = get_result("Angle");
        result.allocate_single_value();
        result.set_float_value(0.0f);
      }
      return;
    }

    MovieClip *movie_clip = get_movie_clip();
    const int frame_number = BKE_movieclip_remap_scene_to_clip_frame(
        movie_clip, context().get_scene()->r.cfra);
    const int width = GPU_texture_width(movie_clip_texture);
    const int height = GPU_texture_height(movie_clip_texture);

    /* If the movie clip has no stabilization data, it will initialize the given values with
     * fallback values regardless, so no need to handle that case. */
    float2 offset;
    float scale, angle;
    BKE_tracking_stabilization_data_get(
        movie_clip, frame_number, width, height, offset, &scale, &angle);

    if (is_output_needed("Offset X")) {
      Result &result = get_result("Offset X");
      result.allocate_single_value();
      result.set_float_value(offset.x);
    }
    if (is_output_needed("Offset Y")) {
      Result &result = get_result("Offset Y");
      result.allocate_single_value();
      result.set_float_value(offset.y);
    }
    if (is_output_needed("Scale")) {
      Result &result = get_result("Scale");
      result.allocate_single_value();
      result.set_float_value(scale);
    }
    if (is_output_needed("Angle")) {
      Result &result = get_result("Angle");
      result.allocate_single_value();
      result.set_float_value(angle);
    }
  }

  GPUTexture *get_movie_clip_texture()
  {
    MovieClip *movie_clip = get_movie_clip();
    MovieClipUser *movie_clip_user = static_cast<MovieClipUser *>(node().storage);
    BKE_movieclip_user_set_frame(movie_clip_user, context().get_scene()->r.cfra);
    return BKE_movieclip_get_gpu_texture(movie_clip, movie_clip_user);
  }

  void free_movie_clip_texture()
  {
    MovieClip *movie_clip = get_movie_clip();
    return BKE_movieclip_free_gputexture(movie_clip);
  }

  MovieClip *get_movie_clip()
  {
    return (MovieClip *)node().id;
  }
};

static NodeOperation *get_compositor_operation(Context &context, DNode node)
{
  return new MovieClipOperation(context, node);
}

}  // namespace blender::nodes::node_composite_movieclip_cc

void register_node_type_cmp_movieclip()
{
  namespace file_ns = blender::nodes::node_composite_movieclip_cc;

  static bNodeType ntype;

  cmp_node_type_base(&ntype, CMP_NODE_MOVIECLIP, "Movie Clip", NODE_CLASS_INPUT);
  ntype.declare = file_ns::cmp_node_movieclip_declare;
  ntype.draw_buttons = file_ns::node_composit_buts_movieclip;
  ntype.draw_buttons_ex = file_ns::node_composit_buts_movieclip_ex;
  ntype.get_compositor_operation = file_ns::get_compositor_operation;
  ntype.initfunc_api = file_ns::init;
  ntype.flag |= NODE_PREVIEW;
  node_type_storage(
      &ntype, "MovieClipUser", node_free_standard_storage, node_copy_standard_storage);

  nodeRegisterType(&ntype);
}
