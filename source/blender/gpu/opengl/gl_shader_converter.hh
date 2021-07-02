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

namespace blender::gpu {

enum class GLShaderConverterState {
  OkUnchanged,
  OkChanged,
  MismatchedPushConstantNames,
};

struct PatchContext {
  struct {
    std::string name;
  } push_constants;
};

class GLShaderConverter {
 public:
  GLShaderConverterState state = GLShaderConverterState::OkUnchanged;

 private:
  Vector<std::string> patched_sources_;
  PatchContext context_;

 public:
  void patch(MutableSpan<const char *> sources);
  bool has_errors() const;
  void free();

  MEM_CXX_CLASS_ALLOC_FUNCS("GLShaderConverter");
};

}  // namespace blender::gpu
