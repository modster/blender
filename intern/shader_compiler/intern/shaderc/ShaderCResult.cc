
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