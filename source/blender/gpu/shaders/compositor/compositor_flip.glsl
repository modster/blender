#pragma BLENDER_REQUIRE(gpu_shader_compositor_texture_utils.glsl)

void main()
{
  ivec2 xy = ivec2(gl_GlobalInvocationID.xy);
  ivec2 size = texture_size(input_image);
  ivec2 flipped_xy = xy;
#if defined(FLIP_X)
  flipped_xy.x = size.x - xy.x - 1;
#endif
#if defined(FLIP_Y)
  flipped_xy.y = size.y - xy.y - 1;
#endif
  imageStore(output_image, xy, texture_load(input_image, flipped_xy));
}
