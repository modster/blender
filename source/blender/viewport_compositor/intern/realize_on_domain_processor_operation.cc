/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

#include "BLI_math_vec_types.hh"
#include "BLI_transformation_2d.hh"
#include "BLI_utildefines.h"

#include "GPU_compute.h"
#include "GPU_shader.h"
#include "GPU_texture.h"

#include "VPC_context.hh"
#include "VPC_domain.hh"
#include "VPC_input_descriptor.hh"
#include "VPC_realize_on_domain_processor_operation.hh"
#include "VPC_result.hh"

namespace blender::viewport_compositor {

RealizeOnDomainProcessorOperation::RealizeOnDomainProcessorOperation(Context &context,
                                                                     Domain domain,
                                                                     ResultType type)
    : ProcessorOperation(context), domain_(domain)
{
  InputDescriptor input_descriptor;
  input_descriptor.type = type;
  declare_input_descriptor(input_descriptor);
  populate_result(Result(type, texture_pool()));
}

void RealizeOnDomainProcessorOperation::execute()
{
  Result &input = get_input();
  Result &result = get_result();

  result.allocate_texture(domain_);

  GPUShader *shader = get_realization_shader();
  GPU_shader_bind(shader);

  /* Transform the input space into the domain space. */
  const Transformation2D local_transformation = input.domain().transformation *
                                                domain_.transformation.inverted();

  /* Set the pivot of the transformation to be the center of the domain.  */
  const float2 pivot = float2(domain_.size) / 2.0f;
  const Transformation2D pivoted_transformation = local_transformation.set_pivot(pivot);

  /* Invert the transformation because the shader transforms the domain coordinates instead of the
   * input image itself and thus expect the inverse. */
  const Transformation2D inverse_transformation = pivoted_transformation.inverted();

  /* Set the inverse of the transform to the shader. */
  GPU_shader_uniform_mat3(shader, "inverse_transformation", inverse_transformation.matrix());

  /* The texture sampler should use bilinear interpolation for both the bilinear and bicubic
   * cases, as the logic used by the bicubic realization shader expects textures to use bilinear
   * interpolation. */
  const bool use_bilinear = ELEM(input.get_realization_options().interpolation,
                                 Interpolation::Bilinear,
                                 Interpolation::Bicubic);
  GPU_texture_filter_mode(input.texture(), use_bilinear);

  /* Make out-of-bound texture access return zero by clamping to border color. And make texture
   * wrap appropriately if the input repeats. */
  const bool repeats = input.get_realization_options().repeat_x ||
                       input.get_realization_options().repeat_y;
  GPU_texture_wrap_mode(input.texture(), repeats, false);

  input.bind_as_texture(shader, "input_sampler");
  result.bind_as_image(shader, "domain");

  const int2 size = result.domain().size;
  GPU_compute_dispatch(shader, size.x / 16 + 1, size.y / 16 + 1, 1);

  input.unbind_as_texture();
  result.unbind_as_image();
  GPU_shader_unbind();
  GPU_shader_free(shader);
}

GPUShader *RealizeOnDomainProcessorOperation::get_realization_shader()
{
  switch (get_result().type()) {
    case ResultType::Color:
      return GPU_shader_create_from_info_name("compositor_realize_on_domain_color");
    case ResultType::Vector:
      return GPU_shader_create_from_info_name("compositor_realize_on_domain_vector");
    case ResultType::Float:
      return GPU_shader_create_from_info_name("compositor_realize_on_domain_float");
  }

  BLI_assert_unreachable();
  return nullptr;
}

Domain RealizeOnDomainProcessorOperation::compute_domain()
{
  return domain_;
}

}  // namespace blender::viewport_compositor
