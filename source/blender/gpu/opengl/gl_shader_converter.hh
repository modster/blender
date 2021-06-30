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
  Ok,
  NOT_MATCHING_PUSH_CONSTANT_NAME,
};
class GLShaderConverter {
 public:
  struct {
    std::string name;
  } push_constants;
  GLShaderConverterState status = GLShaderConverterState::Ok;

 private:
  Vector<std::string> patched_sources_;

 public:
  void patch(MutableSpan<const char *> sources);
  bool has_error() const;
  void free();

 private:
  std::string patch(StringRef src);
  bool is_valid_name_char(const char c) const;
  StringRef skip_whitespace(StringRef src);

  StringRef extract_name(StringRef src);
  std::string patch_push_constants(StringRef src);

  MEM_CXX_CLASS_ALLOC_FUNCS("GLShaderConverter");
};

}  // namespace blender::gpu
