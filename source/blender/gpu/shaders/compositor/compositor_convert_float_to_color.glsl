/* Fill all three color channels of the output with the input and set the alpha channel to 1. */

void main()
{
  ivec2 xy = ivec2(gl_GlobalInvocationID.xy);
  imageStore(output_image, xy, vec4(texelFetch(input_sampler, xy, 0).xxx, 1.0));
}
