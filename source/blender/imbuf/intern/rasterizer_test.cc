/* SPDX-License-Identifier: Apache-2.0 */

#include "testing/testing.h"

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
  void vertex(const VertexInputType &input, VertexOutputType *r_output) override
  {
    r_output->uv = input.uv * image_size;
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
  IMB_initImBuf(&image_buffer, IMBUF_SIZE, IMBUF_SIZE, 0, IB_rectfloat);

  Rasterizer<VertexShader, FragmentShader, 4096, Stats> rasterizer(&image_buffer);

  VertexShader &vertex_shader = rasterizer.vertex_shader();
  vertex_shader.image_size = float2(image_buffer.x, image_buffer.y);

  EXPECT_EQ(rasterizer.stats.triangles, 0);
  EXPECT_EQ(rasterizer.stats.discarded_triangles, 0);
  EXPECT_EQ(rasterizer.stats.rasterlines, 0);
  EXPECT_EQ(rasterizer.stats.discarded_rasterlines, 0);
  EXPECT_EQ(rasterizer.stats.clamped_rasterlines, 0);
  EXPECT_EQ(rasterizer.stats.drawn_fragments, 0);

  rasterizer.draw_triangle(
      VertexInput(float2(0.1, 0.1)), VertexInput(float2(0.5, 0.2)), VertexInput(float2(0.4, 0.9)));
  rasterizer.flush();

  /*
    EXPECT_EQ(rasterizer.stats.triangles, 1);
    EXPECT_EQ(rasterizer.stats.discarded_triangles, 0);
    EXPECT_EQ(rasterizer.stats.rasterlines, 245);
    EXPECT_EQ(rasterizer.stats.discarded_rasterlines, 1);
    EXPECT_EQ(rasterizer.stats.clamped_rasterlines, 0);
    // EXPECT_EQ(rasterizer.stats.drawn_fragments, 0);
  */

  for (int y = 0; y < IMBUF_SIZE; y++) {
    for (int x = 0; x < IMBUF_SIZE; x++) {
      int pixel_offset = y * IMBUF_SIZE + x;
      float *pixel = &image_buffer.rect_float[pixel_offset * 4];
      printf("%s", *pixel < 0.5 ? " " : "#");
    }
    printf("\n");
  }

  imb_freerectImbuf_all(&image_buffer);
}

}  // namespace blender::imbuf::rasterizer::tests
