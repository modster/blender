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
