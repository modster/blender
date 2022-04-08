/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

#pragma once

#include <cstdint>

#include "BLI_math_vec_types.hh"
#include "BLI_transformation_2d.hh"

namespace blender::viewport_compositor {

/* --------------------------------------------------------------------
 * Realization Options.
 */

/* Possible interpolations to use when realizing an input result of some domain on another domain.
 * See the RealizationOptions class for more information. */
enum class Interpolation : uint8_t {
  Nearest,
  Bilinear,
  Bicubic,
};

/* The options that describe how an input result prefer to be realized on some other domain. This
 * is used by the RealizeOnDomainProcessorOperation to identify the appropriate method of
 * realization. See the Domain class for more information. */
class RealizationOptions {
 public:
  /* The interpolation method that should be used when performing realization. Since realizing a
   * result involves projecting it on a different domain, which in turn, involves sampling the
   * result at arbitrary locations, the interpolation identifies the method used for computing the
   * value at those arbitrary locations. */
  Interpolation interpolation = Interpolation::Nearest;
  /* If true, the result will be repeated infinitely along the horizontal axis when realizing the
   * result. If false, regions outside of bounds of the result along the horizontal axis will be
   * filled with zeros. */
  bool repeat_x = false;
  /* If true, the result will be repeated infinitely along the vertical axis when realizing the
   * result. If false, regions outside of bounds of the result along the vertical axis will be
   * filled with zeros. */
  bool repeat_y = false;
};

/* --------------------------------------------------------------------
 * Domain.
 */

/* A domain is a rectangular area of a certain size in pixels that is transformed by a certain
 * transformation in pixel space relative to some reference space.
 *
 * Any result computed by an operation resides in a domain. The size of the domain of the result is
 * the size of its texture. The transformation of the domain of the result is typically an identity
 * transformation, indicating that the result is centered in space. But a transformation operation
 * like the rotate, translate, or transform operations will adjust the transformation to make the
 * result reside somewhere different in space. The domain of a single value result is irrelevant
 * and always set to an identity domain.
 *
 * An operation operates in a certain domain called the operation domain, it follows that the
 * operation only cares about the inputs whose domain is inside or at least intersects the
 * operation domain. To abstract away the different domains of the inputs, any input that have a
 * different domain than the operation domain is realized on the operation domain through a
 * RealizeOnDomainProcessorOperation, except inputs whose descriptor sets skip_realization or
 * expects_single_value, see InputDescriptor for more information. The realization process simply
 * projects the input domain on the operation domain, copies the area of input that intersects the
 * operation domain, and fill the rest with zeros or repetitions of the input domain; depending on
 * the realization_options, see the RealizationOptions class for more information. This process is
 * illustrated below, assuming no repetition in either directions. It follows that operations
 * should expect all their inputs to have the same domain and consequently size, except for inputs
 * that explicitly skip realization.
 *
 *                                   Realized Result
 *             +-------------+       +-------------+
 *             |  Operation  |       |             |
 *             |   Domain    |       |    Zeros    |
 *             |             | ----> |             |
 *       +-----------+       |       |-----+       |
 *       |     |  C  |       |       |  C  |       |
 *       |     +-----|-------+       +-----|-------+
 *       | Domain Of |
 *       |   Input   |
 *       +-----------+
 *
 * Each operation can define an arbitrary operation domain, but in most cases, the operation domain
 * is inferred from the inputs. By default, the operation domain is computed as follows. Typically,
 * one input of the operation is said to be the domain input and the operation domain is inferred
 * from it. The domain input is determined to be the non-single value input that have the highest
 * domain priority, a zero value being the highest priority. If all inputs are single values, then
 * the operation domain is irrelevant and an identity domain is set. See
 * NodeOperation::compute_domain for more information.
 *
 * The aforementioned logic for operation domain computation is only a default that works for most
 * cases, but an operation can override the compute_domain method to implement a different logic.
 * For instance, output nodes have an operation domain the same size as the viewport and with an
 * identity transformation, their operation domain doesn't depend on the inputs at all.
 *
 * For instance, a filter operation have two inputs, a factor and a color, the latter of which
 * has a domain priority of 0 and the former has a domain priority of 1. If the color input is not
 * a single value, then the domain of this operation is computed to be the same size and
 * transformation as the color input, because it has the highest priority. And if the factor input
 * have a different size and/or transformation from the computed domain of the operation, it will
 * be projected and realized on it to have the same size as described above. It follows that the
 * color input, will not need to be realized because it already has the same size and
 * transformation as the domain of the operation, because the operation domain is inferred from it.
 * On the other hand, if the color input is a single value input, then the operation domain will be
 * the same as the domain of the factor input, because it has the second highest domain priority.
 * Finally, if both inputs are single value inputs, the operation domain will be an identity and is
 * irrelevant. */
class Domain {
 public:
  /* The size of the domain in pixels. */
  int2 size;
  /* The 2D transformation of the domain defining its translation in pixels, rotation, and scale in
   * 2D space. */
  Transformation2D transformation;
  /* The options that describe how this domain prefer to be realized on some other domain. See the
   * RealizationOptions for more information. */
  RealizationOptions realization_options;

 public:
  /* A size only constructor that sets the transformation to identity. */
  Domain(int2 size);

  Domain(int2 size, Transformation2D transformation);

  /* Transform the domain by the given transformation. This effectively pre-multiply the given
   * transformation by the current transformation of the domain. */
  void transform(const Transformation2D &transformation);

  /* Returns a domain of size 1x1 and an identity transformation. */
  static Domain identity();
};

/* Compare the size and transformation of the domain. The realization_options are not compared
 * because they only describe the method of realization on another domain, which is not technically
 * a proprty of the domain itself. */
bool operator==(const Domain &a, const Domain &b);

/* Inverse of the above equality operator. */
bool operator!=(const Domain &a, const Domain &b);

}  // namespace blender::viewport_compositor
