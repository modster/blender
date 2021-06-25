/* Apache License, Version 2.0 */

#include "gpu_testing.hh"
#include "shader_compiler.hh"

namespace blender::gpu::tests {

#ifdef WITH_VULKAN
TEST_F(GPUVulkanTest, shader_compiler)
{
  std::string source = "#version 450\nvoid main() {}";
  shader_compiler::Compiler *compiler = shader_compiler::Compiler::create_default();
  shader_compiler::Job job;
  job.source = source;
  job.name = __func__;
  job.source_type = shader_compiler::SourceType::GlslComputeShader;
  job.compilation_target = shader_compiler::TargetType::SpirV;

  shader_compiler::Result *result = compiler->compile(job);
  EXPECT_EQ(result->type, shader_compiler::TargetType::SpirV);
  EXPECT_EQ(result->status_code, shader_compiler::StatusCode::Ok);
  EXPECT_GT(result->bin.size(), 0);
  EXPECT_EQ(result->error_log, "");
  delete result;
  delete compiler;
}
#endif

}  // namespace blender::gpu::tests