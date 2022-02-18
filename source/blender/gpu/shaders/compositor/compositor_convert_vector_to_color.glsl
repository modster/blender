/* Copy the three vector components of the input to the three color channels of the output and set
 * the alpha channel to 1. */

void main()
{
  ivec2 xy = ivec2(gl_GlobalInvocationID.xy);
  imageStore(output_image, xy, vec4(texelFetch(input_sampler, xy, 0).xyz, 1.0));
}
