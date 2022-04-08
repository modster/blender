/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

#pragma once

#include "BLI_math_vec_types.hh"
#include "BLI_transformation_2d.hh"

#include "GPU_shader.h"
#include "GPU_texture.h"

#include "VPC_domain.hh"
#include "VPC_texture_pool.hh"

namespace blender::viewport_compositor {

/* Possible data types that operations can operate on. They either represent the base type of the
 * result texture or a single value result. */
enum class ResultType : uint8_t {
  Float,
  Vector,
  Color,
};

/* A class that represents an output of an operation. A result reside in a certain domain defined
 * by its size and transformation, see the Domain class for more information. A result either
 * stores a single value or a texture. An operation will output a single value result if that value
 * would have been constant over the whole texture. Single value results are stored in 1x1 textures
 * to make them easily accessible in shaders. But the same value is also stored in the value member
 * of the result for any host-side processing. */
class Result {
 private:
  /* The base type of the texture or the type of the single value. */
  ResultType type_;
  /* If true, the result is a single value, otherwise, the result is a texture. */
  bool is_single_value_;
  /* A GPU texture storing the result data. This will be a 1x1 texture if the result is a single
   * value, the value of which will be identical to that of the value member. See class description
   * for more information. */
  GPUTexture *texture_ = nullptr;
  /* The texture pool used to allocate the texture of the result, this should be initialized during
   * construction. */
  TexturePool *texture_pool_ = nullptr;
  /* The number of users currently referencing and using this result. If this result have a master
   * result, then this reference count is irrelevant and shadowed by the reference count of the
   * master result. */
  int reference_count_ = 0;
  /* If the result is a single value, this member stores the value of the result, the value of
   * which will be identical to that stored in the texture member. While this member stores 4
   * values, only a subset of which could be initialized depending on the type, for instance, a
   * float result will only initialize the first array element and a vector result will only
   * initialize the first three array elements. This member is uninitialized if the result is a
   * texture. */
  float value_[4];
  /* The domain of the result. This only matters if the result was a texture. See the Domain class
   * for more information. */
  Domain domain_ = Domain::identity();
  /* If not nullptr, then this result wraps and uses the texture of another master result. In this
   * case, calls to texture-related methods like increment_reference_count and release should
   * operate on the master result as opposed to this result. This member is typically set upon
   * calling the pass_through method, which sets this result to be the master of a target result.
   * See that method for more information. */
  Result *master_ = nullptr;

 public:
  /* Construct a result of the given type with the given texture pool that will be used to allocate
   * and release the result's texture. */
  Result(ResultType type, TexturePool &texture_pool);

  /* Declare the result to be a texture result, allocate a texture of an appropriate type with
   * the size of the given domain from the result's texture pool, and set the domain of the result
   * to the given domain. */
  void allocate_texture(Domain domain);

  /* Declare the result to be a single value result, allocate a texture of an appropriate
   * type with size 1x1 from the result's texture pool, and set the domain to be an identity
   * domain. See class description for more information. */
  void allocate_single_value();

  /* Bind the texture of the result to the texture image unit with the given name in the currently
   * bound given shader. This also inserts a memory barrier for texture fetches to ensure any prior
   * writes to the texture are reflected before reading from it. */
  void bind_as_texture(GPUShader *shader, const char *texture_name) const;

  /* Bind the texture of the result to the image unit with the given name in the currently bound
   * given shader. */
  void bind_as_image(GPUShader *shader, const char *image_name) const;

  /* Unbind the texture which was previously bound using bind_as_texture. */
  void unbind_as_texture() const;

  /* Unbind the texture which was previously bound using bind_as_image. */
  void unbind_as_image() const;

  /* Pass this result through to a target result. This method makes the target result a copy of
   * this result, essentially having identical values between the two and consequently sharing the
   * underlying texture. Additionally, this result is set to be the master of the target result, by
   * setting the master member of the target. Finally, the reference count of the result is
   * incremented by the reference count of the target result. This is typically called in the
   * allocate method of an operation whose input texture will not change and can be passed to the
   * output directly. It should be noted that such operations can still adjust other properties of
   * the result, like its domain. So for instance, the transform operation passes its input through
   * to its output because it will not change it, however, it may adjusts its domain. */
  void pass_through(Result &target);

  /* Transform the result by the given transformation. This effectively pre-multiply the given
   * transformation by the current transformation of the domain of the result. */
  void transform(const Transformation2D &transformation);

  /* Get a reference to the realization options of this result. See the RealizationOptions class
   * for more information. */
  RealizationOptions &get_realization_options();

  /* If the result is a single value result of type float, return its float value. Otherwise, an
   * uninitialized value is returned. */
  float get_float_value() const;

  /* If the result is a single value result of type vector, return its vector value. Otherwise, an
   * uninitialized value is returned. */
  float3 get_vector_value() const;

  /* If the result is a single value result of type color, return its color value. Otherwise, an
   * uninitialized value is returned. */
  float4 get_color_value() const;

  /* Same as get_float_value but returns a default value if the result is not a single value. */
  float get_float_value_default(float default_value) const;

  /* Same as get_vector_value but returns a default value if the result is not a single value. */
  float3 get_vector_value_default(const float3 &default_value) const;

  /* Same as get_color_value but returns a default value if the result is not a single value. */
  float4 get_color_value_default(const float4 &default_value) const;

  /* If the result is a single value result of type float, set its float value and upload it to the
   * texture. Otherwise, an undefined behavior is invoked. */
  void set_float_value(float value);

  /* If the result is a single value result of type vector, set its vector value and upload it to
   * the texture. Otherwise, an undefined behavior is invoked. */
  void set_vector_value(const float3 &value);

  /* If the result is a single value result of type color, set its color value and upload it to the
   * texture. Otherwise, an undefined behavior is invoked. */
  void set_color_value(const float4 &value);

  /* Increment the reference count of the result by the given count. This should be called when a
   * user gets a reference to the result to use. If this result have a master result, the reference
   * count of the master result is incremented instead. */
  void increment_reference_count(int count = 1);

  /* Decrement the reference count of the result and release the result texture back into the
   * texture pool if the reference count reaches zero. This should be called when a user that
   * previously referenced and incremented the reference count of the result no longer needs it. If
   * this result have a master result, the master result is released instead. */
  void release();

  /* Returns the type of the result. */
  ResultType type() const;

  /* Returns true if the result is a texture and false of it is a single value. */
  bool is_texture() const;

  /* Returns true if the result is a single value and false of it is a texture. */
  bool is_single_value() const;

  /* Returns the allocated GPU texture of the result. */
  GPUTexture *texture() const;

  /* Returns the reference count of the result. If this result have a master result, then the
   * reference count of the master result is returned instead. */
  int reference_count() const;

  /* Returns a reference to the domain of the result. See the Domain class. */
  const Domain &domain() const;
};

}  // namespace blender::viewport_compositor
