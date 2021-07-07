#include "testing/testing.h"

#include "GPU_uniform_buffer_types.h"
#include "gpu_uniform_buffer_private.hh"

#include "BLI_math.h"

namespace blender::gpu::tests {

TEST(GPUUniformStruct, struct1)
{
  UniformBuiltinStruct uniform_struct(GPU_UNIFORM_STRUCT_1);
  const UniformBuiltinStructType &type_info = uniform_struct.type_info();
  const GPUUniformBuiltinStruct1 *struct_data = static_cast<const GPUUniformBuiltinStruct1 *>(
      uniform_struct.data());
  EXPECT_EQ(type_info.data_size(), sizeof(*struct_data));

  /* ModelMatrix attribute. */
  {
    const UniformBuiltinStructType::AttributeBinding &binding = type_info.attribute_binding(
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
    const UniformBuiltinStructType::AttributeBinding &binding = type_info.attribute_binding(
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
    const UniformBuiltinStructType::AttributeBinding &binding = type_info.attribute_binding(
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
    const UniformBuiltinStructType::AttributeBinding &binding = type_info.attribute_binding(
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
    const UniformBuiltinStructType::AttributeBinding &binding = type_info.attribute_binding(
        GPU_UNIFORM_SRGB_TRANSFORM);
    int srgb_transform = true;
    const bool result = uniform_struct.uniform_int(binding.binding, 1, 1, &srgb_transform);
    EXPECT_TRUE(result);
    EXPECT_EQ(struct_data->SrgbTransform, srgb_transform);
  }
}

}  // namespace blender::gpu::tests