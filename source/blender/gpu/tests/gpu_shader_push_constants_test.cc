/* Apache License, Version 2.0 */

#include "testing/testing.h"

#include "GPU_shader.h"
#include "GPU_uniform_buffer.h"

#include "gpu_testing.hh"

namespace blender::gpu::tests {

struct PushConstants {
  float color[4];
};

static void test_gpu_shader_push_constants()
{
  const char *vert_glsl = R"(

uniform mat4 ModelViewProjectionMatrix;
in vec3 pos;

void main() {
  vec4 pos_4d = vec4(pos, 1.0);
  gl_Position = ModelViewProjectionMatrix * pos_4d;
}

)";

  const char *frag_glsl = R"(

layout(push_constant) uniform PushConstants {
    vec4 color;
};

out vec4 fragColor;

void main()
{
  fragColor = color;
}

)";

  GPUShader *shader = GPU_shader_create(
      vert_glsl, frag_glsl, nullptr, nullptr, nullptr, "test_gpu_shader_push_constants");
  EXPECT_NE(shader, nullptr);

  PushConstants push_constants;
  GPUUniformBuf *push_constants_buffer = GPU_uniformbuf_create_ex(
      sizeof(PushConstants), &push_constants, __func__);
  GPU_shader_uniform_push_constant(shader, push_constants_buffer);
  GPU_uniformbuf_free(push_constants_buffer);

  GPU_shader_free(shader);
}
GPU_TEST(gpu_shader_push_constants)

static void test_gpu_shader_push_constants_2_definitions()
{
  const char *vert_glsl = R"(

uniform mat4 ModelViewProjectionMatrix;
in vec3 pos;

void main() {
  vec4 pos_4d = vec4(pos, 1.0);
  gl_Position = ModelViewProjectionMatrix * pos_4d;
}

)";

  const char *frag_glsl = R"(

#ifdef NEW
layout(push_constant) uniform PushConstants {
    vec4 color;
};
#else 
layout(push_constant) uniform PushConstants {
    vec4 color;
};
#endif

out vec4 fragColor;

void main()
{
  fragColor = color;
}

)";

  GPUShader *shader = GPU_shader_create(
      vert_glsl, frag_glsl, nullptr, nullptr, nullptr, "gpu_shader_push_constants_2_definitions");
  EXPECT_NE(shader, nullptr);

  GPU_shader_free(shader);
}
GPU_TEST(gpu_shader_push_constants_2_definitions)

static void test_gpu_shader_push_constants_2_stages()
{
  const char *vert_glsl = R"(

uniform mat4 ModelViewProjectionMatrix;
in vec3 pos;

layout(push_constant) uniform PushConstants {
    vec4 color;
};

void main() {
  vec4 pos_4d = vec4(pos, 1.0);
  gl_Position = ModelViewProjectionMatrix * pos_4d;
}

)";

  const char *frag_glsl = R"(

layout(push_constant) uniform PushConstants {
    vec4 color;
};

out vec4 fragColor;

void main()
{
  fragColor = color;
}

)";

  GPUShader *shader = GPU_shader_create(
      vert_glsl, frag_glsl, nullptr, nullptr, nullptr, "test_gpu_shader_push_constants_2_stages");
  EXPECT_NE(shader, nullptr);

  GPU_shader_free(shader);
}
GPU_TEST(gpu_shader_push_constants_2_stages)

static void test_gpu_shader_push_constants_fail_different_name_2_stages()
{
  const char *vert_glsl = R"(

uniform mat4 ModelViewProjectionMatrix;
in vec3 pos;

layout(push_constant) uniform PushConstantsVert {
    vec4 color;
};

void main() {
  vec4 pos_4d = vec4(pos, 1.0);
  gl_Position = ModelViewProjectionMatrix * pos_4d;
}

)";

  const char *frag_glsl = R"(

layout(push_constant) uniform PushConstantsFrag {
    vec4 color;
};

out vec4 fragColor;

void main()
{
  fragColor = color;
}

)";

  GPUShader *shader = GPU_shader_create(vert_glsl,
                                        frag_glsl,
                                        nullptr,
                                        nullptr,
                                        nullptr,
                                        "gpu_shader_push_constants_fail_different_name_2_stages");
  /* Compiling should fail as two differnet binding names are used within a single shader. */
  EXPECT_EQ(shader, nullptr);

  GPU_shader_free(shader);
}
GPU_TEST(gpu_shader_push_constants_fail_different_name_2_stages)

static void test_gpu_shader_push_constants_fail_different_name_1_stage()
{
  const char *vert_glsl = R"(

uniform mat4 ModelViewProjectionMatrix;
flat out vec4 finalColor;
in vec3 pos;

void main() {
  vec4 pos_4d = vec4(pos, 1.0);
  gl_Position = ModelViewProjectionMatrix * pos_4d;
}

)";

  const char *frag_glsl = R"(

#if 0
layout(push_constant) uniform PushConstants_new {
    vec4 color;
};
#else 
layout(push_constant) uniform PushConstants_old {
    vec4 color;
};
#endif

out vec4 fragColor;

void main()
{
  fragColor = color;
}

)";

  GPUShader *shader = GPU_shader_create(vert_glsl,
                                        frag_glsl,
                                        nullptr,
                                        nullptr,
                                        nullptr,
                                        "gpu_shader_push_constants_fail_different_name_1_stage");
  /* Compiling should fail as two differnet binding names are used within a single shader. */
  EXPECT_EQ(shader, nullptr);

  GPU_shader_free(shader);
}
GPU_TEST(gpu_shader_push_constants_fail_different_name_1_stage)

}  // namespace blender::gpu::tests
