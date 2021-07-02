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
 */

#pragma once

#include "BLI_string_ref.hh"
#include "BLI_vector.hh"

#include <optional>

namespace blender::gpu {

enum class GLShaderPatcherState {
  OkUnchanged,
  OkChanged,
  MismatchedPushConstantNames,
};

/** State to keep over GLSL compilation stages, linkage and shader_interface building. */
struct GLShaderPatcherContext {
  GLShaderPatcherState state = GLShaderPatcherState::OkUnchanged;

  /**
   * All patched sources. During compilation stage source code is references as `const
   * char*` These needs to be owned by a `std::string`.
   */
  Vector<std::string> patched_sources;
  struct {
    std::optional<std::string> name;
  } push_constants;

  bool has_errors() const;

  void free_patched_sources();
};

void patch_vulkan_to_opengl(GLShaderPatcherContext &context, MutableSpan<const char *> sources);

}  // namespace blender::gpu
