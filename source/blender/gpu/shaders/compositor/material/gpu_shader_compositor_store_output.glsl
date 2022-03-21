void store_output_float(restrict writeonly image2D output_image, float value)
{
  ivec2 xy = ivec2(gl_GlobalInvocationID.xy);
  imageStore(output_image, xy, vec4(value));
}

void store_output_vector(restrict writeonly image2D output_image, vec3 vector)
{
  ivec2 xy = ivec2(gl_GlobalInvocationID.xy);
  imageStore(output_image, xy, vec4(vector, 0.0));
}

void store_output_color(restrict writeonly image2D output_image, vec4 color)
{
  ivec2 xy = ivec2(gl_GlobalInvocationID.xy);
  imageStore(output_image, xy, color);
}
