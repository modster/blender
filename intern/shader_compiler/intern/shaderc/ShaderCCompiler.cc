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

#include "ShaderCCompiler.hh"
#include "ShaderCResult.hh"
#include "shaderc/shaderc.hpp"

namespace shader_compiler::shaderc {

ShaderCCompiler::ShaderCCompiler()
{
}

ShaderCCompiler::~ShaderCCompiler()
{
}

void ShaderCCompiler::set_optimization_level(::shaderc::CompileOptions options,
                                             const OptimizationLevel new_value)
{
  switch (new_value) {
    case OptimizationLevel::NotOptimized:
      options.SetOptimizationLevel(shaderc_optimization_level_zero);
      break;
    case OptimizationLevel::SizeOptimized:
      options.SetOptimizationLevel(shaderc_optimization_level_size);
      break;
    case OptimizationLevel::SpeedOptimized:
      options.SetOptimizationLevel(shaderc_optimization_level_performance);
      break;
  }
}

shaderc_shader_kind ShaderCCompiler::get_source_kind(SourceType source_type)
{
  switch (source_type) {
    case SourceType::GlslVertexShader:
      return shaderc_glsl_vertex_shader;

    case SourceType::GlslGeometryShader:
      return shaderc_glsl_geometry_shader;

    case SourceType::GlslFragmentShader:
      return shaderc_glsl_fragment_shader;

    case SourceType::GlslComputeShader:
      return shaderc_glsl_compute_shader;
  }
  return shaderc_glsl_vertex_shader;
}

ShaderCResult *ShaderCCompiler::compile_spirv(const Job &job)
{
  ::shaderc::CompileOptions options;
  set_optimization_level(options, job.optimization_level);
  shaderc_shader_kind kind = get_source_kind(job.source_type);

  ::shaderc::SpvCompilationResult shaderc_result = compiler_.CompileGlslToSpv(
      job.source.c_str(), kind, job.name.c_str(), options);

  ShaderCResult *result = new ShaderCResult();
  result->init(job, shaderc_result);
  return result;
}

Result *ShaderCCompiler::compile(const Job &job)
{
  switch (job.compilation_target) {
    case TargetType::SpirV:
      return compile_spirv(job);
      break;
  }
  return nullptr;
}

}  // namespace shader_compiler::shaderc