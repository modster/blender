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
 * The Original Code is Copyright (C) 2021 by Blender Foundation.
 * All rights reserved.
 */

#pragma once

#include "shader_compiler.hh"

#include "shaderc/shaderc.hpp"

namespace shader_compiler::shaderc {

class ShaderCCompiler : public Compiler {
 private:
  ::shaderc::Compiler compiler_;

 public:
  ShaderCCompiler();
  ~ShaderCCompiler() override;

  Result *compile(const Job &job) override;

 private:
  static void set_optimization_level(::shaderc::CompileOptions options,
                                     const OptimizationLevel new_value);
  static shaderc_shader_kind get_source_kind(SourceType source_type);
};

}  // namespace shader_compiler::shaderc
