/* The shader create info instance should define a CONVERT_EXPRESSION as well as an output_image
 * given the sampled texel. */
void main()
{
  ivec2 xy = ivec2(gl_GlobalInvocationID.xy);
  vec4 texel = texelFetch(input_sampler, xy, 0);
  imageStore(output_image, xy, CONVERT_EXPRESSION);
}
