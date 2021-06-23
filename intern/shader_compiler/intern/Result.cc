#include "shader_compiler.hh"

namespace shader_compiler {

void Result::init(const Job &job)
{
  type = job.compilation_target;
}

}  // namespace shader_compiler
