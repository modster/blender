#pragma BLENDER_REQUIRE(gpu_shader_compositor_texture_utils.glsl)

void main()
{
  ivec2 xy = ivec2(gl_GlobalInvocationID.xy);
#if defined(SPLIT_HORIZONTAL)
  bool condition = (view_size.x * split_ratio) < xy.x;
#elif defined(SPLIT_VERTICAL)
  bool condition = (view_size.y * split_ratio) < xy.y;
#endif
  vec4 color = condition ? load_texture(first_image, xy) : load_texture(second_image, xy);
  imageStore(output_image, xy, color);
}
