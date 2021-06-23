
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

Result *ShaderCCompiler::compile(const Job &job)
{
  ::shaderc::CompileOptions options;
  set_optimization_level(options, job.optimization_level);

  shaderc_shader_kind kind = get_source_kind(job.source_type);

  ::shaderc::SpvCompilationResult shaderc_result = compiler_.CompileGlslToSpv(
      job.source, kind, job.name, options);

  ShaderCResult *result = new ShaderCResult();
  result->init(job, shaderc_result);
  return result;
}

}  // namespace shader_compiler::shaderc