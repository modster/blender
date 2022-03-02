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
    r_output->data = float4(input.value, input.value, input.value, 1.0);
  }
};

class FragmentShader : public AbstractFragmentShader<float4, float4> {
 public:
  void fragment(const FragmentInputType &input, FragmentOutputType *r_output) override
  {
    *r_output = input;
  }
};

/* Draw 2 triangles that fills the entire image buffer and see if each pixel is touched. */
TEST(imbuf_rasterizer, draw_triangle_edge_alignment_quality)
{
  ImBuf image_buffer;
  IMB_initImBuf(&image_buffer, IMBUF_SIZE, IMBUF_SIZE, 32, IB_rectfloat);

  Rasterizer<VertexShader, FragmentShader, DefaultRasterlinesBufferSize, Stats> rasterizer(
      &image_buffer);

  VertexShader &vertex_shader = rasterizer.vertex_shader();
  vertex_shader.image_size = float2(image_buffer.x, image_buffer.y);

  float clear_color[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  float3 location(0.5, 0.5, 0.0);
  float3 rotation(0.0, 0.0, 0.0);
  float3 scale(1.0, 1.0, 1.0);

  for (int i = 0; i < 1000; i++) {
    rasterizer.stats.reset();


    IMB_rectfill(&image_buffer, clear_color);
    rotation[2] = (i / 1000.0) * M_PI * 2;

    vertex_shader.vp_mat = float4x4::from_loc_eul_scale(location, rotation, scale);
    rasterizer.draw_triangle(VertexInput(float2(-1.0, -1.0), 0.2),
                             VertexInput(float2(-1.0, 1.0), 0.5),
                             VertexInput(float2(1.0, -1.0), 1.0));
    rasterizer.draw_triangle(VertexInput(float2(1.0, 1.0), 0.2),
                             VertexInput(float2(-1.0, 1.0), 0.5),
                             VertexInput(float2(1.0, -1.0), 1.0));
    rasterizer.flush();

    /* Check if each pixel has been drawn exactly once. */
    EXPECT_EQ(rasterizer.stats.drawn_fragments, IMBUF_SIZE * IMBUF_SIZE) <<  i;

#ifdef DEBUG_SAVE
  char file_name[FILE_MAX];
    BLI_path_sequence_encode(file_name, "/tmp/test_", ".png", 4, i);
    IMB_saveiff(&image_buffer, file_name, IB_rectfloat);
    imb_freerectImBuf(&image_buffer);
#endif

  }

  imb_freerectImbuf_all(&image_buffer);
}
/**
 * This test case renders 3 images that should have the same coverage. But using a different edge.
 *
 * The results should be identical.
 */
TEST(imbuf_rasterizer, edge_pixel_clamping)
{
  using RasterizerType =
      Rasterizer<VertexShader, FragmentShader, DefaultRasterlinesBufferSize, Stats>;
  float clear_color[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  ImBuf image_buffer_a;
  ImBuf image_buffer_b;
  ImBuf image_buffer_c;
  int fragments_drawn_a;
  int fragments_drawn_b;
  int fragments_drawn_c;

  {
    IMB_initImBuf(&image_buffer_a, IMBUF_SIZE, IMBUF_SIZE, 32, IB_rectfloat);

    RasterizerType rasterizer_a(&image_buffer_a);
    VertexShader &vertex_shader = rasterizer_a.vertex_shader();
    vertex_shader.image_size = float2(image_buffer_a.x, image_buffer_a.y);
    vertex_shader.vp_mat = float4x4::identity();
    IMB_rectfill(&image_buffer_a, clear_color);
    rasterizer_a.draw_triangle(VertexInput(float2(0.2, -0.2), 1.0),
                               VertexInput(float2(1.2, 1.2), 1.0),
                               VertexInput(float2(1.5, -0.3), 1.0));
    rasterizer_a.flush();
    fragments_drawn_a = rasterizer_a.stats.drawn_fragments;
  }
  {
    IMB_initImBuf(&image_buffer_b, IMBUF_SIZE, IMBUF_SIZE, 32, IB_rectfloat);

    RasterizerType rasterizer_b(&image_buffer_b);
    VertexShader &vertex_shader = rasterizer_b.vertex_shader();
    vertex_shader.image_size = float2(image_buffer_b.x, image_buffer_b.y);
    vertex_shader.vp_mat = float4x4::identity();
    IMB_rectfill(&image_buffer_b, clear_color);
    rasterizer_b.draw_triangle(VertexInput(float2(0.2, -0.2), 1.0),
                               VertexInput(float2(1.2, 1.2), 1.0),
                               VertexInput(float2(1.5, -0.3), 1.0));
    rasterizer_b.flush();
    fragments_drawn_b = rasterizer_b.stats.drawn_fragments;
  }

  {
    IMB_initImBuf(&image_buffer_c, IMBUF_SIZE, IMBUF_SIZE, 32, IB_rectfloat);

    RasterizerType rasterizer_c(&image_buffer_c);
    VertexShader &vertex_shader = rasterizer_c.vertex_shader();
    vertex_shader.image_size = float2(image_buffer_c.x, image_buffer_c.y);
    vertex_shader.vp_mat = float4x4::identity();
    IMB_rectfill(&image_buffer_c, clear_color);
    rasterizer_c.draw_triangle(VertexInput(float2(0.2, -0.2), 1.0),
                               VertexInput(float2(1.2, 1.2), 1.0),
                               VertexInput(float2(10.0, 1.3), 1.0));
    rasterizer_c.flush();
    fragments_drawn_c = rasterizer_c.stats.drawn_fragments;
  }

  EXPECT_EQ(fragments_drawn_a, fragments_drawn_b);
  EXPECT_EQ(memcmp(image_buffer_a.rect_float,
                   image_buffer_b.rect_float,
                   sizeof(float) * 4 * IMBUF_SIZE * IMBUF_SIZE),
            0);
  EXPECT_EQ(fragments_drawn_a, fragments_drawn_c);
  EXPECT_EQ(memcmp(image_buffer_a.rect_float,
                   image_buffer_c.rect_float,
                   sizeof(float) * 4 * IMBUF_SIZE * IMBUF_SIZE),
            0);
  EXPECT_EQ(fragments_drawn_b, fragments_drawn_c);
  EXPECT_EQ(memcmp(image_buffer_b.rect_float,
                   image_buffer_c.rect_float,
                   sizeof(float) * 4 * IMBUF_SIZE * IMBUF_SIZE),
            0);

  imb_freerectImbuf_all(&image_buffer_a);
  imb_freerectImbuf_all(&image_buffer_b);
  imb_freerectImbuf_all(&image_buffer_c);
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
