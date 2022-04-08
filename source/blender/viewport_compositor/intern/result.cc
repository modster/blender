/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

#include "BLI_math_vec_types.hh"
#include "BLI_transformation_2d.hh"

#include "GPU_shader.h"
#include "GPU_state.h"
#include "GPU_texture.h"

#include "VPC_domain.hh"
#include "VPC_result.hh"
#include "VPC_texture_pool.hh"

namespace blender::viewport_compositor {

Result::Result(ResultType type, TexturePool &texture_pool)
    : type_(type), texture_pool_(&texture_pool)
{
}

void Result::allocate_texture(Domain domain)
{
  is_single_value_ = false;
  switch (type_) {
    case ResultType::Float:
      texture_ = texture_pool_->acquire_float(domain.size);
      break;
    case ResultType::Vector:
      texture_ = texture_pool_->acquire_vector(domain.size);
      break;
    case ResultType::Color:
      texture_ = texture_pool_->acquire_color(domain.size);
      break;
  }
  domain_ = domain;
}

void Result::allocate_single_value()
{
  is_single_value_ = true;
  /* Single values are stored in 1x1 textures. */
  const int2 texture_size{1, 1};
  switch (type_) {
    case ResultType::Float:
      texture_ = texture_pool_->acquire_float(texture_size);
      break;
    case ResultType::Vector:
      texture_ = texture_pool_->acquire_vector(texture_size);
      break;
    case ResultType::Color:
      texture_ = texture_pool_->acquire_color(texture_size);
      break;
  }
  domain_ = Domain::identity();
}

void Result::bind_as_texture(GPUShader *shader, const char *texture_name) const
{
  /* Make sure any prior writes to the texture are reflected before reading from it. */
  GPU_memory_barrier(GPU_BARRIER_TEXTURE_FETCH);

  const int texture_image_unit = GPU_shader_get_texture_binding(shader, texture_name);
  GPU_texture_bind(texture_, texture_image_unit);
}

void Result::bind_as_image(GPUShader *shader, const char *image_name) const
{
  const int image_unit = GPU_shader_get_texture_binding(shader, image_name);
  GPU_texture_image_bind(texture_, image_unit);
}

void Result::unbind_as_texture() const
{
  GPU_texture_unbind(texture_);
}

void Result::unbind_as_image() const
{
  GPU_texture_image_unbind(texture_);
}

void Result::pass_through(Result &target)
{
  /* Increment the reference count of the master by the original reference count of the target. */
  increment_reference_count(target.reference_count());
  /* Copy the result to the target and set its master. */
  target = *this;
  target.master_ = this;
}

void Result::transform(const Transformation2D &transformation)
{
  domain_.transform(transformation);
}

RealizationOptions &Result::get_realization_options()
{
  return domain_.realization_options;
}

float Result::get_float_value() const
{
  return *value_;
}

float3 Result::get_vector_value() const
{
  return float3(value_);
}

float4 Result::get_color_value() const
{
  return float4(value_);
}

float Result::get_float_value_default(float default_value) const
{
  if (is_single_value()) {
    return get_float_value();
  }
  return default_value;
}

float3 Result::get_vector_value_default(const float3 &default_value) const
{
  if (is_single_value()) {
    return get_vector_value();
  }
  return default_value;
}

float4 Result::get_color_value_default(const float4 &default_value) const
{
  if (is_single_value()) {
    return get_color_value();
  }
  return default_value;
}

void Result::set_float_value(float value)
{
  *value_ = value;
  GPU_texture_update(texture_, GPU_DATA_FLOAT, value_);
}

void Result::set_vector_value(const float3 &value)
{
  copy_v3_v3(value_, value);
  GPU_texture_update(texture_, GPU_DATA_FLOAT, value_);
}

void Result::set_color_value(const float4 &value)
{
  copy_v4_v4(value_, value);
  GPU_texture_update(texture_, GPU_DATA_FLOAT, value_);
}

void Result::increment_reference_count(int count)
{
  /* If there is a master result, increment its reference count instead. */
  if (master_) {
    master_->increment_reference_count(count);
    return;
  }

  reference_count_ += count;
}

void Result::release()
{
  /* If there is a master result, release it instead. */
  if (master_) {
    master_->release();
    return;
  }

  /* Decrement the reference count, and if it reaches zero, release the texture back into the
   * texture pool. */
  reference_count_--;
  if (reference_count_ == 0) {
    texture_pool_->release(texture_);
  }
}

ResultType Result::type() const
{
  return type_;
}

bool Result::is_texture() const
{
  return !is_single_value_;
}

bool Result::is_single_value() const
{
  return is_single_value_;
}

GPUTexture *Result::texture() const
{
  return texture_;
}

int Result::reference_count() const
{
  /* If there is a master result, return its reference count instead. */
  if (master_) {
    return master_->reference_count();
  }
  return reference_count_;
}

const Domain &Result::domain() const
{
  return domain_;
}

}  // namespace blender::viewport_compositor
