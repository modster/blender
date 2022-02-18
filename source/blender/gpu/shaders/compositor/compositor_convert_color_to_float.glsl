/* Store the average of the input's three color channels in the output, the alpha channel is
 * ignored. */

void main()
{
  ivec2 xy = ivec2(gl_GlobalInvocationID.xy);
  vec4 color = texelFetch(input_sampler, xy, 0);
  imageStore(output_image, xy, vec4((color.r + color.g + color.b) / 3.0));
}
