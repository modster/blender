/* Apache License, Version 2.0 */

#include "gpu_testing.hh"

#include "GPU_shader.h"

namespace blender::gpu::tests {

static void test_compile_builtin_shader(eGPUBuiltinShader shader_type, eGPUShaderConfig sh_cfg)
{
  GPUShader *sh = GPU_shader_get_builtin_shader_with_config(shader_type, sh_cfg);
  EXPECT_NE(sh, nullptr);
}

TEST_F(GPUOpenGLTest, shader_builtin_opengl)
{
  GPU_shader_free_builtin_shaders();
  test_compile_builtin_shader(GPU_SHADER_3D_DEPTH_ONLY, GPU_SHADER_CFG_DEFAULT);
}

#ifdef WITH_VULKAN

TEST_F(GPUVulkanTest, shader_builtin_vulkan)
{
  GPU_shader_free_builtin_shaders();
  test_compile_builtin_shader(GPU_SHADER_3D_DEPTH_ONLY, GPU_SHADER_CFG_DEFAULT);
}

#endif

}  // namespace blender::gpu::tests