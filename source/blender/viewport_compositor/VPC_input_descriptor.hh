/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

#pragma once

#include "VPC_result.hh"

namespace blender::viewport_compositor {

/* A class that describes an input of an operation. */
class InputDescriptor {
 public:
  /* The type of input. This may be different that the type of result that the operation will
   * receive for the input, in which case, an implicit conversion input processor operation will
   * be added to convert it to the required type. */
  ResultType type;
  /* If true, then the input does not need to be realized on the domain of the operation before its
   * execution. See the Domain class for more information. */
  bool skip_realization = false;
  /* The priority of the input for determining the operation domain. The non-single value input
   * with the highest priority will be used to infer the operation domain, the highest priority
   * being zero. See the Domain class for more information. */
  int domain_priority = 0;
  /* If true, the input expects a single value, and if a non-single value is provided, a default
   * single value will be used instead, see the get_*_value_default methods in the Result
   * class. It follows that this also imply skip_realization, because we don't need to realize a
   * result that will be discarded anyways. If false, the input can work with both single and
   * non-single values. */
  bool expects_single_value = false;
};

}  // namespace blender::viewport_compositor
