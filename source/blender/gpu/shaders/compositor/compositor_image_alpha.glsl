/* Copy the alpha of the input float 2D sampler to the output R16F 2D image. */

void main()
{
  ivec2 xy = ivec2(gl_GlobalInvocationID.xy);
  imageStore(output_image, xy, texelFetch(input_sampler, xy, 0).aaaa);
}
