/* SPDX-License-Identifier: Apache-2.0 */

#include "testing/testing.h"

#include "BLI_float4x4.hh"
#include "BLI_path_util.h"

#include "IMB_rasterizer.hh"

namespace blender::imbuf::rasterizer::tests {

const uint32_t IMBUF_SIZE = 128;

struct VertexInput {
  float2 uv;

  VertexInput(float2 uv) : uv(uv)
  {
  }
};

class VertexShader : public AbstractVertexShader<VertexInput, float> {
 public:
  float2 image_size;
  float4x4 vp_mat;
  void vertex(const VertexInputType &input, VertexOutputType *r_output) override
  {
    float3 t = float3(input.uv[0], input.uv[1], 0.0);
    r_output->coord = float2(vp_mat * t) * image_size;
    r_output->data = 1.0f;
  }
};

class FragmentShader : public AbstractFragmentShader<float, float4> {
 public:
  void fragment(const FragmentInputType &input, FragmentOutputType *r_output) override
  {
    *r_output = float4(input, input, input, 1.0);
  }
};

TEST(imbuf_rasterizer, draw_triangle)
{
  ImBuf image_buffer;
  IMB_initImBuf(&image_buffer, IMBUF_SIZE, IMBUF_SIZE, 32, IB_rectfloat);

  Rasterizer<VertexShader, FragmentShader, DefaultRasterlinesBufferSize, Stats> rasterizer(
      &image_buffer);

  VertexShader &vertex_shader = rasterizer.vertex_shader();
  vertex_shader.image_size = float2(image_buffer.x, image_buffer.y);

  EXPECT_EQ(rasterizer.stats.triangles, 0);
  EXPECT_EQ(rasterizer.stats.discarded_triangles, 0);
  EXPECT_EQ(rasterizer.stats.rasterlines, 0);
  EXPECT_EQ(rasterizer.stats.discarded_rasterlines, 0);
  EXPECT_EQ(rasterizer.stats.clamped_rasterlines, 0);
  EXPECT_EQ(rasterizer.stats.drawn_fragments, 0);

  float clear_color[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  char file_name[FILE_MAX];

  float3 location(0.5, 0.5, 0.0);
  float3 rotation(0.0, 0.0, 0.0);
  float3 scale(1.0, 1.0, 1.0);

  for (int i = 0; i < 1000; i++) {
    BLI_path_sequence_encode(file_name, "/tmp/test_", ".png", 4, i);
    printf("%s: %s\n", __func__, file_name);

    if (i == 43) {
      printf("break\n");
    }

    IMB_rectfill(&image_buffer, clear_color);
    rotation[2] = (i / 1000.0) * M_PI * 2;

    vertex_shader.vp_mat = float4x4::from_loc_eul_scale(location, rotation, scale);
    rasterizer.draw_triangle(VertexInput(float2(-0.4, -0.4)),
                             VertexInput(float2(0.0, 0.4)),
                             VertexInput(float2(0.3, 0.0)));
    rasterizer.flush();

    IMB_saveiff(&image_buffer, file_name, IB_rectfloat);
    imb_freerectImBuf(&image_buffer);
  }

  imb_freerectImbuf_all(&image_buffer);
}

}  // namespace blender::imbuf::rasterizer::tests
