#pragma BLENDER_REQUIRE(gpu_shader_compositor_texture_utils.glsl)

void main()
{
  ivec2 xy = ivec2(gl_GlobalInvocationID.xy);

  /* First, transform the input image by transforming the domain coordinates with the inverse of
   * input image's transformation. The inverse transformation is an affine matrix and thus the
   * coordinates should be in homogeneous coordinates.  */
  vec2 coordinates = (inverse_transformation * vec3(xy, 1.0)).xy;

  /* Since an input image with an identity transformation is supposed to be centered in the domain,
   * we subtract the offset between the lower left corners of the input image and the domain, which
   * is half the difference between their sizes, because the difference in size is on both sides of
   * the centered image. */
  ivec2 domain_size = imageSize(domain);
  ivec2 input_size = texture_size(input_sampler);
  vec2 offset = (domain_size - input_size) / 2.0;

  /* Subtract the offset and divide by the input image size to get the relevant coordinates into
   * the sampler's expected [0, 1] range. */
  vec2 normalized_coordinates = (coordinates - offset) / input_size;

  imageStore(domain, xy, texture(input_sampler, normalized_coordinates));
}
