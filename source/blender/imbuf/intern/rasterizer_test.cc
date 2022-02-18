/* SPDX-License-Identifier: Apache-2.0 */

#include "testing/testing.h"

#include "BLI_float4x4.hh"
#include "BLI_path_util.h"

#include "IMB_rasterizer.hh"

namespace blender::imbuf::rasterizer::tests {

const uint32_t IMBUF_SIZE = 256;

struct VertexInput {
  float2 uv;
  float value;

  VertexInput(float2 uv, float value) : uv(uv), value(value)
  {
  }
};

class VertexShader : public AbstractVertexShader<VertexInput, float4> {
 public:
  float2 image_size;
  float4x4 vp_mat;
  void vertex(const VertexInputType &input, VertexOutputType *r_output) override
  {
    float2 coord = float2(vp_mat * float3(input.uv[0], input.uv[1], 0.0));
    r_output->coord = coord * image_size;
    r_output->data = float4(coord[0], coord[1], input.value, 1.0);
  }
};

class FragmentShader : public AbstractFragmentShader<float4, float4> {
 public:
  void fragment(const FragmentInputType &input, FragmentOutputType *r_output) override
  {
    *r_output = input;
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

    IMB_rectfill(&image_buffer, clear_color);
    rotation[2] = (i / 1000.0) * M_PI * 2;

    vertex_shader.vp_mat = float4x4::from_loc_eul_scale(location, rotation, scale);
    rasterizer.draw_triangle(VertexInput(float2(-0.4, -0.4), 0.2),
                             VertexInput(float2(0.0, 0.4), 0.5),
                             VertexInput(float2(0.3, 0.0), 1.0));
    rasterizer.flush();

    IMB_saveiff(&image_buffer, file_name, IB_rectfloat);
    imb_freerectImBuf(&image_buffer);
  }

  imb_freerectImbuf_all(&image_buffer);
}

TEST(imbuf_rasterizer, center_pixel_clamper_scanline_for)
{
  CenterPixelClampingMethod clamper;

  EXPECT_EQ(clamper.scanline_for(-2.0f), -2);
  EXPECT_EQ(clamper.scanline_for(-1.9f), -2);
  EXPECT_EQ(clamper.scanline_for(-1.8f), -2);
  EXPECT_EQ(clamper.scanline_for(-1.7f), -2);
  EXPECT_EQ(clamper.scanline_for(-1.6f), -2);
  EXPECT_EQ(clamper.scanline_for(-1.5f), -2);
  EXPECT_EQ(clamper.scanline_for(-1.4f), -1);
  EXPECT_EQ(clamper.scanline_for(-1.3f), -1);
  EXPECT_EQ(clamper.scanline_for(-1.2f), -1);
  EXPECT_EQ(clamper.scanline_for(-1.1f), -1);
  EXPECT_EQ(clamper.scanline_for(-1.0f), -1);
  EXPECT_EQ(clamper.scanline_for(-0.9f), -1);
  EXPECT_EQ(clamper.scanline_for(-0.8f), -1);
  EXPECT_EQ(clamper.scanline_for(-0.7f), -1);
  EXPECT_EQ(clamper.scanline_for(-0.6f), -1);
  EXPECT_EQ(clamper.scanline_for(-0.5f), -1);
  EXPECT_EQ(clamper.scanline_for(-0.4f), 0);
  EXPECT_EQ(clamper.scanline_for(-0.3f), 0);
  EXPECT_EQ(clamper.scanline_for(-0.2f), 0);
  EXPECT_EQ(clamper.scanline_for(-0.1f), 0);
  EXPECT_EQ(clamper.scanline_for(0.0f), 0);
  EXPECT_EQ(clamper.scanline_for(0.1f), 0);
  EXPECT_EQ(clamper.scanline_for(0.2f), 0);
  EXPECT_EQ(clamper.scanline_for(0.3f), 0);
  EXPECT_EQ(clamper.scanline_for(0.4f), 0);
  EXPECT_EQ(clamper.scanline_for(0.5f), 0);
  EXPECT_EQ(clamper.scanline_for(0.6f), 1);
  EXPECT_EQ(clamper.scanline_for(0.7f), 1);
  EXPECT_EQ(clamper.scanline_for(0.8f), 1);
  EXPECT_EQ(clamper.scanline_for(0.9f), 1);
  EXPECT_EQ(clamper.scanline_for(1.0f), 1);
  EXPECT_EQ(clamper.scanline_for(1.0f), 1);
  EXPECT_EQ(clamper.scanline_for(1.1f), 1);
  EXPECT_EQ(clamper.scanline_for(1.2f), 1);
  EXPECT_EQ(clamper.scanline_for(1.3f), 1);
  EXPECT_EQ(clamper.scanline_for(1.4f), 1);
  EXPECT_EQ(clamper.scanline_for(1.5f), 1);
  EXPECT_EQ(clamper.scanline_for(1.6f), 2);
  EXPECT_EQ(clamper.scanline_for(1.7f), 2);
  EXPECT_EQ(clamper.scanline_for(1.8f), 2);
  EXPECT_EQ(clamper.scanline_for(1.9f), 2);
  EXPECT_EQ(clamper.scanline_for(2.0f), 2);
}

TEST(imbuf_rasterizer, center_pixel_clamper_column_for)
{
  CenterPixelClampingMethod clamper;

  EXPECT_EQ(clamper.column_for(-2.0f), -2);
  EXPECT_EQ(clamper.column_for(-1.9f), -2);
  EXPECT_EQ(clamper.column_for(-1.8f), -2);
  EXPECT_EQ(clamper.column_for(-1.7f), -2);
  EXPECT_EQ(clamper.column_for(-1.6f), -2);
  EXPECT_EQ(clamper.column_for(-1.5f), -2);
  EXPECT_EQ(clamper.column_for(-1.4f), -1);
  EXPECT_EQ(clamper.column_for(-1.3f), -1);
  EXPECT_EQ(clamper.column_for(-1.2f), -1);
  EXPECT_EQ(clamper.column_for(-1.1f), -1);
  EXPECT_EQ(clamper.column_for(-1.0f), -1);
  EXPECT_EQ(clamper.column_for(-0.9f), -1);
  EXPECT_EQ(clamper.column_for(-0.8f), -1);
  EXPECT_EQ(clamper.column_for(-0.7f), -1);
  EXPECT_EQ(clamper.column_for(-0.6f), -1);
  EXPECT_EQ(clamper.column_for(-0.5f), -1);
  EXPECT_EQ(clamper.column_for(-0.4f), 0);
  EXPECT_EQ(clamper.column_for(-0.3f), 0);
  EXPECT_EQ(clamper.column_for(-0.2f), 0);
  EXPECT_EQ(clamper.column_for(-0.1f), 0);
  EXPECT_EQ(clamper.column_for(0.0f), 0);
  EXPECT_EQ(clamper.column_for(0.1f), 0);
  EXPECT_EQ(clamper.column_for(0.2f), 0);
  EXPECT_EQ(clamper.column_for(0.3f), 0);
  EXPECT_EQ(clamper.column_for(0.4f), 0);
  EXPECT_EQ(clamper.column_for(0.5f), 0);
  EXPECT_EQ(clamper.column_for(0.6f), 1);
  EXPECT_EQ(clamper.column_for(0.7f), 1);
  EXPECT_EQ(clamper.column_for(0.8f), 1);
  EXPECT_EQ(clamper.column_for(0.9f), 1);
  EXPECT_EQ(clamper.column_for(1.0f), 1);
  EXPECT_EQ(clamper.column_for(1.0f), 1);
  EXPECT_EQ(clamper.column_for(1.1f), 1);
  EXPECT_EQ(clamper.column_for(1.2f), 1);
  EXPECT_EQ(clamper.column_for(1.3f), 1);
  EXPECT_EQ(clamper.column_for(1.4f), 1);
  EXPECT_EQ(clamper.column_for(1.5f), 1);
  EXPECT_EQ(clamper.column_for(1.6f), 2);
  EXPECT_EQ(clamper.column_for(1.7f), 2);
  EXPECT_EQ(clamper.column_for(1.8f), 2);
  EXPECT_EQ(clamper.column_for(1.9f), 2);
  EXPECT_EQ(clamper.column_for(2.0f), 2);
}

TEST(imbuf_rasterizer, center_pixel_clamper_distance_to_scanline_anchorpoint)
{
  CenterPixelClampingMethod clamper;

  EXPECT_FLOAT_EQ(clamper.distance_to_scanline_anchor(-2.0f), 0.5f);
  EXPECT_FLOAT_EQ(clamper.distance_to_scanline_anchor(-1.9f), 0.4f);
  EXPECT_FLOAT_EQ(clamper.distance_to_scanline_anchor(-1.8f), 0.3f);
  EXPECT_FLOAT_EQ(clamper.distance_to_scanline_anchor(-1.7f), 0.2f);
  EXPECT_FLOAT_EQ(clamper.distance_to_scanline_anchor(-1.6f), 0.1f);
  EXPECT_FLOAT_EQ(clamper.distance_to_scanline_anchor(-1.5f), 0.0f);
  EXPECT_FLOAT_EQ(clamper.distance_to_scanline_anchor(-1.4f), 0.9f);
  EXPECT_FLOAT_EQ(clamper.distance_to_scanline_anchor(-1.3f), 0.8f);
  EXPECT_FLOAT_EQ(clamper.distance_to_scanline_anchor(-1.2f), 0.7f);
  EXPECT_FLOAT_EQ(clamper.distance_to_scanline_anchor(-1.1f), 0.6f);
  EXPECT_FLOAT_EQ(clamper.distance_to_scanline_anchor(-1.0f), 0.5f);
  EXPECT_FLOAT_EQ(clamper.distance_to_scanline_anchor(-0.9f), 0.4f);
  EXPECT_FLOAT_EQ(clamper.distance_to_scanline_anchor(-0.8f), 0.3f);
  EXPECT_FLOAT_EQ(clamper.distance_to_scanline_anchor(-0.7f), 0.2f);
  EXPECT_FLOAT_EQ(clamper.distance_to_scanline_anchor(-0.6f), 0.1f);
  EXPECT_FLOAT_EQ(clamper.distance_to_scanline_anchor(-0.5f), 0.0f);
  EXPECT_FLOAT_EQ(clamper.distance_to_scanline_anchor(-0.4f), 0.9f);
  EXPECT_FLOAT_EQ(clamper.distance_to_scanline_anchor(-0.3f), 0.8f);
  EXPECT_FLOAT_EQ(clamper.distance_to_scanline_anchor(-0.2f), 0.7f);
  EXPECT_FLOAT_EQ(clamper.distance_to_scanline_anchor(-0.1f), 0.6f);
  EXPECT_FLOAT_EQ(clamper.distance_to_scanline_anchor(0.0f), 0.5f);
  EXPECT_FLOAT_EQ(clamper.distance_to_scanline_anchor(0.1f), 0.4f);
  EXPECT_FLOAT_EQ(clamper.distance_to_scanline_anchor(0.2f), 0.3f);
  EXPECT_FLOAT_EQ(clamper.distance_to_scanline_anchor(0.3f), 0.2f);
  EXPECT_FLOAT_EQ(clamper.distance_to_scanline_anchor(0.4f), 0.1f);
  EXPECT_FLOAT_EQ(clamper.distance_to_scanline_anchor(0.5f), 0.0f);
  EXPECT_FLOAT_EQ(clamper.distance_to_scanline_anchor(0.6f), 0.9f);
  EXPECT_FLOAT_EQ(clamper.distance_to_scanline_anchor(0.7f), 0.8f);
  EXPECT_FLOAT_EQ(clamper.distance_to_scanline_anchor(0.8f), 0.7f);
  EXPECT_FLOAT_EQ(clamper.distance_to_scanline_anchor(0.9f), 0.6f);
  EXPECT_FLOAT_EQ(clamper.distance_to_scanline_anchor(1.0f), 0.5f);
  EXPECT_FLOAT_EQ(clamper.distance_to_scanline_anchor(1.1f), 0.4f);
  EXPECT_FLOAT_EQ(clamper.distance_to_scanline_anchor(1.2f), 0.3f);
  EXPECT_FLOAT_EQ(clamper.distance_to_scanline_anchor(1.3f), 0.2f);
  EXPECT_FLOAT_EQ(clamper.distance_to_scanline_anchor(1.4f), 0.1f);
  EXPECT_FLOAT_EQ(clamper.distance_to_scanline_anchor(1.5f), 0.0f);
  EXPECT_FLOAT_EQ(clamper.distance_to_scanline_anchor(1.6f), 0.9f);
  EXPECT_FLOAT_EQ(clamper.distance_to_scanline_anchor(1.7f), 0.8f);
  EXPECT_FLOAT_EQ(clamper.distance_to_scanline_anchor(1.8f), 0.7f);
  EXPECT_FLOAT_EQ(clamper.distance_to_scanline_anchor(1.9f), 0.6f);
  EXPECT_FLOAT_EQ(clamper.distance_to_scanline_anchor(2.0f), 0.5f);
}

TEST(imbuf_rasterizer, center_pixel_clamper_distance_to_column_anchorpoint)
{
  CenterPixelClampingMethod clamper;

  EXPECT_FLOAT_EQ(clamper.distance_to_column_anchor(-2.0f), 0.5f);
  EXPECT_FLOAT_EQ(clamper.distance_to_column_anchor(-1.9f), 0.4f);
  EXPECT_FLOAT_EQ(clamper.distance_to_column_anchor(-1.8f), 0.3f);
  EXPECT_FLOAT_EQ(clamper.distance_to_column_anchor(-1.7f), 0.2f);
  EXPECT_FLOAT_EQ(clamper.distance_to_column_anchor(-1.6f), 0.1f);
  EXPECT_FLOAT_EQ(clamper.distance_to_column_anchor(-1.5f), 0.0f);
  EXPECT_FLOAT_EQ(clamper.distance_to_column_anchor(-1.4f), 0.9f);
  EXPECT_FLOAT_EQ(clamper.distance_to_column_anchor(-1.3f), 0.8f);
  EXPECT_FLOAT_EQ(clamper.distance_to_column_anchor(-1.2f), 0.7f);
  EXPECT_FLOAT_EQ(clamper.distance_to_column_anchor(-1.1f), 0.6f);
  EXPECT_FLOAT_EQ(clamper.distance_to_column_anchor(-1.0f), 0.5f);
  EXPECT_FLOAT_EQ(clamper.distance_to_column_anchor(-0.9f), 0.4f);
  EXPECT_FLOAT_EQ(clamper.distance_to_column_anchor(-0.8f), 0.3f);
  EXPECT_FLOAT_EQ(clamper.distance_to_column_anchor(-0.7f), 0.2f);
  EXPECT_FLOAT_EQ(clamper.distance_to_column_anchor(-0.6f), 0.1f);
  EXPECT_FLOAT_EQ(clamper.distance_to_column_anchor(-0.5f), 0.0f);
  EXPECT_FLOAT_EQ(clamper.distance_to_column_anchor(-0.4f), 0.9f);
  EXPECT_FLOAT_EQ(clamper.distance_to_column_anchor(-0.3f), 0.8f);
  EXPECT_FLOAT_EQ(clamper.distance_to_column_anchor(-0.2f), 0.7f);
  EXPECT_FLOAT_EQ(clamper.distance_to_column_anchor(-0.1f), 0.6f);
  EXPECT_FLOAT_EQ(clamper.distance_to_column_anchor(0.0f), 0.5f);
  EXPECT_FLOAT_EQ(clamper.distance_to_column_anchor(0.1f), 0.4f);
  EXPECT_FLOAT_EQ(clamper.distance_to_column_anchor(0.2f), 0.3f);
  EXPECT_FLOAT_EQ(clamper.distance_to_column_anchor(0.3f), 0.2f);
  EXPECT_FLOAT_EQ(clamper.distance_to_column_anchor(0.4f), 0.1f);
  EXPECT_FLOAT_EQ(clamper.distance_to_column_anchor(0.5f), 0.0f);
  EXPECT_FLOAT_EQ(clamper.distance_to_column_anchor(0.6f), 0.9f);
  EXPECT_FLOAT_EQ(clamper.distance_to_column_anchor(0.7f), 0.8f);
  EXPECT_FLOAT_EQ(clamper.distance_to_column_anchor(0.8f), 0.7f);
  EXPECT_FLOAT_EQ(clamper.distance_to_column_anchor(0.9f), 0.6f);
  EXPECT_FLOAT_EQ(clamper.distance_to_column_anchor(1.0f), 0.5f);
  EXPECT_FLOAT_EQ(clamper.distance_to_column_anchor(1.1f), 0.4f);
  EXPECT_FLOAT_EQ(clamper.distance_to_column_anchor(1.2f), 0.3f);
  EXPECT_FLOAT_EQ(clamper.distance_to_column_anchor(1.3f), 0.2f);
  EXPECT_FLOAT_EQ(clamper.distance_to_column_anchor(1.4f), 0.1f);
  EXPECT_FLOAT_EQ(clamper.distance_to_column_anchor(1.5f), 0.0f);
  EXPECT_FLOAT_EQ(clamper.distance_to_column_anchor(1.6f), 0.9f);
  EXPECT_FLOAT_EQ(clamper.distance_to_column_anchor(1.7f), 0.8f);
  EXPECT_FLOAT_EQ(clamper.distance_to_column_anchor(1.8f), 0.7f);
  EXPECT_FLOAT_EQ(clamper.distance_to_column_anchor(1.9f), 0.6f);
  EXPECT_FLOAT_EQ(clamper.distance_to_column_anchor(2.0f), 0.5f);
}

}  // namespace blender::imbuf::rasterizer::tests
