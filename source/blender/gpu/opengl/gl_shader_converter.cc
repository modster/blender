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
 * The Original Code is Copyright (C) 2021 Blender Foundation.
 * All rights reserved.
 */

/** \file
 * \ingroup gpu
 *
 * Parse/convert GLSL source from Vulkan GLSL to OpenGL GLSL.
 */

#include "gl_shader_converter.hh"

namespace blender::gpu {

void GLShaderConverter::patch(MutableSpan<const char *> sources)
{
  for (int i = 0; i < sources.size(); i++) {
    std::string patched_source = patch(sources[i]);
    patched_sources_.append(patched_source);
    sources[i] = patched_sources_.last().c_str();
  }
}

bool GLShaderConverter::has_error() const
{
  return status != GLShaderConverterState::Ok;
}

void GLShaderConverter::free()
{
  patched_sources_.clear();
}

std::string GLShaderConverter::patch(StringRef src)
{
  std::string result = patch_push_constants(src);
  return result;
}

StringRef GLShaderConverter::skip_whitespace(StringRef ref)
{
  static constexpr StringRef WHITESPACES = " \t\n\v\f\r";

  size_t skip = ref.find_first_not_of(WHITESPACES);
  if (skip == blender::StringRef::not_found) {
    return ref;
  }
  return ref.drop_prefix(skip);
}

StringRef GLShaderConverter::extract_name(StringRef src)
{
  static constexpr StringRef VALID_CHARS =
      "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ01232456789_";
  StringRef result = src;

  size_t skip = result.find_first_not_of(VALID_CHARS);
  BLI_assert(skip != StringRef::not_found);
  return result.substr(0, skip);
}

std::string GLShaderConverter::patch_push_constants(StringRef src)
{
  static constexpr StringRef LAYOUT_PUSH_CONSTANTS = "layout(push_constant)";
  static constexpr StringRef LAYOUT_STD140 = "layout(std140)";

  size_t pos = src.find(LAYOUT_PUSH_CONSTANTS);
  if (pos == StringRef::not_found) {
    return src;
  }
  std::stringstream result;
  result << src.substr(0, pos);
  result << LAYOUT_STD140;
  result << src.substr(pos + LAYOUT_PUSH_CONSTANTS.size());

  StringRef name = src.substr(pos + LAYOUT_PUSH_CONSTANTS.size());
  name = skip_whitespace(name);
  name = name.drop_known_prefix("uniform");
  name = skip_whitespace(name);
  name = extract_name(name);

  if (push_constants.name.empty()) {
    push_constants.name = name;
  }
  else {
    /* Push constant name must be the same across all stages. */
    if (push_constants.name != name) {
      status = GLShaderConverterState::NOT_MATCHING_PUSH_CONSTANT_NAME;
    }
  }

  return patch_push_constants(result.str());
}

}  // namespace blender::gpu
