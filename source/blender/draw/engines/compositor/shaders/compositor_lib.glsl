/* Keep in sync with CompositorData in compositor_engine.cc. */
struct CompositorData {
  vec3 luminance_coefficients;
  float frame_number;
};

layout(std140) uniform compositor_block
{
  CompositorData compositor_data;
};
