/* SPDX-License-Identifier: GPL-2.0-or-later */

#include "VPC_context.hh"
#include "VPC_shader_pool.hh"
#include "VPC_texture_pool.hh"

namespace blender::viewport_compositor {

Context::Context(TexturePool &texture_pool) : texture_pool_(texture_pool)
{
}

int Context::get_frame_number() const
{
  return get_scene()->r.cfra;
}

float Context::get_time() const
{
  const float frame_number = static_cast<float>(get_frame_number());
  const float frame_rate = static_cast<float>(get_scene()->r.frs_sec) /
                           static_cast<float>(get_scene()->r.frs_sec_base);
  return frame_number / frame_rate;
}

TexturePool &Context::texture_pool()
{
  return texture_pool_;
}

ShaderPool &Context::shader_pool()
{
  return shader_pool_;
}

}  // namespace blender::viewport_compositor
