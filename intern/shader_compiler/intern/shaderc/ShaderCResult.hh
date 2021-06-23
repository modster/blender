#pragma once

#include "shader_compiler.hh"

#include "shaderc/shaderc.hpp"

namespace shader_compiler::shaderc {

class ShaderCResult : public Result {

 public:
  void init(const Job &job, ::shaderc::SpvCompilationResult &shaderc_result);

 private:
  StatusCode status_code_from(::shaderc::SpvCompilationResult &shaderc_result);
  std::string error_log_from(::shaderc::SpvCompilationResult &shaderc_result);
  std::vector<uint32_t> bin_from(::shaderc::SpvCompilationResult &shaderc_result);
};

}  // namespace shader_compiler::shaderc
