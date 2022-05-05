#pragma BLENDER_REQUIRE(gpu_shader_compositor_texture_utilities.glsl)

void main()
{
  ivec2 xy = ivec2(gl_GlobalInvocationID.xy);

  /* Get the normalized coordinates of the pixel centers.  */
  vec2 normalized_xy = (vec2(xy) + vec2(0.5)) / vec2(texture_size(input_image));

  /* Sample the red and blue channels shifted by the dispersion amount. */
  const float red = texture(input_image, normalized_xy + vec2(dispersion, 0.0)).r;
  const float green = texture_load(input_image, xy).g;
  const float blue = texture(input_image, normalized_xy - vec2(dispersion, 0.0)).b;

  imageStore(output_image, xy, vec4(red, green, blue, 1.0));
}
