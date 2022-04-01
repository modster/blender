#pragma BLENDER_REQUIRE(gpu_shader_compositor_texture_utils.glsl)

void main()
{
  ivec2 xy = ivec2(gl_GlobalInvocationID.xy);
  /* Write the pixel color if it is inside the cropping region, otherwise, write zero. */
  bool is_inside = all(greaterThan(xy, lower_bound)) && all(lessThan(xy, higher_bound));
  vec4 color = is_inside ? texture_load(input_image, xy) : vec4(0.0);
  imageStore(output_image, xy, color);
}
