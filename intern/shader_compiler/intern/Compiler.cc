#include "shader_compiler.hh"

#include "intern/shaderc/ShaderCCompiler.hh"

namespace shader_compiler {

Compiler *Compiler::create_default()
{
  Compiler *compiler = new shaderc::ShaderCCompiler();
  return compiler;
}

}  // namespace shader_compiler
