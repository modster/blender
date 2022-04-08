/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

#include "BLI_math_vec_types.hh"
#include "BLI_transformation_2d.hh"

#include "VPC_domain.hh"

namespace blender::viewport_compositor {

Domain::Domain(int2 size) : size(size), transformation(Transformation2D::identity())
{
}

Domain::Domain(int2 size, Transformation2D transformation)
    : size(size), transformation(transformation)
{
}

void Domain::transform(const Transformation2D &input_transformation)
{
  transformation = input_transformation * transformation;
}

Domain Domain::identity()
{
  return Domain(int2(1), Transformation2D::identity());
}

bool operator==(const Domain &a, const Domain &b)
{
  return a.size == b.size && a.transformation == b.transformation;
}

bool operator!=(const Domain &a, const Domain &b)
{
  return !(a == b);
}

}  // namespace blender::viewport_compositor
