#include "gpu_testing.hh"

#include "GPU_capabilities.h"
#include "GPU_compute.h"
#include "GPU_shader.h"
#include "GPU_shader_block_types.h"
#include "gpu_shader_block.hh"

#include "BLI_math.h"

namespace blender::gpu::tests {

TEST(GPUUniformStruct, struct1)
{
  ShaderBlockBuffer uniform_struct(GPU_SHADER_BLOCK_3D_COLOR);
  const ShaderBlockType &type_info = uniform_struct.type_info();
  const GPUShaderBlock3dColor *struct_data = static_cast<const GPUShaderBlock3dColor *>(
      uniform_struct.data());
  EXPECT_EQ(type_info.data_size(), sizeof(*struct_data));

  /* ModelMatrix attribute. */
  {
    const ShaderBlockType::AttributeBinding &binding = type_info.attribute_binding(
        GPU_UNIFORM_MODEL);
    float m4[4][4];
    unit_m4(m4);

    const bool result = uniform_struct.uniform_float(binding.binding, 4, 4, (const float *)m4);
    EXPECT_TRUE(result);
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        EXPECT_EQ(struct_data->ModelMatrix[i][j], m4[i][j]);
      }
    }
  }

  /* ModelViewProjectionMatrix attribute. */
  {
    const ShaderBlockType::AttributeBinding &binding = type_info.attribute_binding(
        GPU_UNIFORM_MVP);
    float m4[4][4];
    unit_m4(m4);

    const bool result = uniform_struct.uniform_float(binding.binding, 4, 4, (const float *)m4);
    EXPECT_TRUE(result);
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        EXPECT_EQ(struct_data->ModelViewProjectionMatrix[i][j], m4[i][j]);
      }
    }
  }

  /* Color attribute. */
  {
    const ShaderBlockType::AttributeBinding &binding = type_info.attribute_binding(
        GPU_UNIFORM_COLOR);
    float color[4] = {1.0f, 0.0f, 1.0f, 1.0f};
    const bool result = uniform_struct.uniform_float(binding.binding, 4, 1, color);
    EXPECT_TRUE(result);
    for (int i = 0; i < 4; i++) {
      EXPECT_EQ(struct_data->color[i], color[i]);
    }
  }

  /* WorldClipPlanes attribute. */
  {
    const ShaderBlockType::AttributeBinding &binding = type_info.attribute_binding(
        GPU_UNIFORM_CLIPPLANES);

    float clip_planes[6][4] = {
        {01.0f, 02.0f, 03.0f, 04.0f},
        {11.0f, 12.0f, 13.0f, 14.0f},
        {21.0f, 22.0f, 23.0f, 24.0f},
        {31.0f, 32.0f, 33.0f, 34.0f},
        {41.0f, 42.0f, 43.0f, 44.0f},
        {51.0f, 52.0f, 53.0f, 54.0f},
    };

    const bool result = uniform_struct.uniform_float(
        binding.binding, 4, 6, (const float *)clip_planes);
    EXPECT_TRUE(result);
    for (int i = 0; i < 6; i++) {
      for (int j = 0; j < 4; j++) {
        EXPECT_EQ(struct_data->WorldClipPlanes[i][j], clip_planes[i][j]);
      }
    }
  }

  /* SrgbTransform attribute. */
  {
    const ShaderBlockType::AttributeBinding &binding = type_info.attribute_binding(
        GPU_UNIFORM_SRGB_TRANSFORM);
    int srgb_transform = true;
    const bool result = uniform_struct.uniform_int(binding.binding, 1, 1, &srgb_transform);
    EXPECT_TRUE(result);
    EXPECT_EQ(struct_data->SrgbTransform, srgb_transform);
  }
}

static void test_custom_shader_with_uniform_builtin_struct()
{
  if (!GPU_compute_shader_support()) {
    /* We can't test as a the platform does not support compute shaders. */
    std::cout << "Skipping compute shader test: platform not supported";
    return;
  }

  /* Build compute shader. */
  const char *compute_glsl = R"(

layout(local_size_x = 1, local_size_y = 1) in;
layout(rgba32f, binding = 0) uniform image2D img_output;

layout(std140) uniform shaderBlock {
  mat4 ModelMatrix;
  mat4 ModelViewProjectionMatrix;
  vec4 color;
  vec4 WorldClipPlanes[6];
  bool SrgbTransform;
};

void main() {
}

)";

  GPUShader *shader = GPU_shader_create_ex(nullptr,
                                           nullptr,
                                           nullptr,
                                           compute_glsl,
                                           nullptr,
                                           nullptr,
                                           GPU_SHADER_TFB_NONE,
                                           nullptr,
                                           0,
                                           GPU_SHADER_BLOCK_3D_COLOR,
                                           __func__);
  EXPECT_NE(shader, nullptr);

  float color[4] = {1.0f, 0.0f, 1.0f, 1.0f};
  GPU_shader_uniform_4fv(shader, "color", color);

  GPU_shader_free(shader);
}

GPU_TEST(custom_shader_with_uniform_builtin_struct)

}  // namespace blender::gpu::tests