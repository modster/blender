/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

#include "VPC_context.hh"
#include "VPC_texture_pool.hh"

namespace blender::viewport_compositor {

Context::Context(TexturePool &texture_pool) : texture_pool_(texture_pool)
{
}

TexturePool &Context::texture_pool()
{
  return texture_pool_;
}

}  // namespace blender::viewport_compositor
