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

#include "ShaderCResult.hh"

namespace shader_compiler::shaderc {

void ShaderCResult::init(const Job &job, ::shaderc::SpvCompilationResult &shaderc_result)
{
  Result::init(job);
  status_code = status_code_from(shaderc_result);

  switch (status_code) {
    case StatusCode::Ok:
      bin = {shaderc_result.cbegin(), shaderc_result.cend()};
      break;

    case StatusCode::CompilationError:
      error_log = shaderc_result.GetErrorMessage();
      break;
  }
}

StatusCode ShaderCResult::status_code_from(::shaderc::SpvCompilationResult &shaderc_result)
{
  if (shaderc_result.GetCompilationStatus() != shaderc_compilation_status_success) {
    return StatusCode::CompilationError;
  }
  return StatusCode::Ok;
}

std::vector<uint32_t> ShaderCResult::bin_from(::shaderc::SpvCompilationResult &shaderc_result)
{
  return {shaderc_result.cbegin(), shaderc_result.cend()};
}

std::string ShaderCResult::error_log_from(::shaderc::SpvCompilationResult &shaderc_result)
{
  return shaderc_result.GetErrorMessage();
}

}  // namespace shader_compiler::shaderc