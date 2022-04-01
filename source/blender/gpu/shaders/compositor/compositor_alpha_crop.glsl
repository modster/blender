#pragma BLENDER_REQUIRE(gpu_shader_compositor_texture_utils.glsl)

void main()
{
  ivec2 xy = ivec2(gl_GlobalInvocationID.xy);
  /* The lower bound is inclusive and upper bound is exclusive. */
  bool is_inside = all(greaterThanEqual(xy, lower_bound)) && all(lessThan(xy, upper_bound));
  /* Write the pixel color if it is inside the cropping region, otherwise, write zero. */
  vec4 color = is_inside ? texture_load(input_image, xy) : vec4(0.0);
  imageStore(output_image, xy, color);
}
