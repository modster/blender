void load_input_float(sampler2D input_sampler, out float value)
{
  ivec2 xy = ivec2(gl_GlobalInvocationID.xy);
  ivec2 texture_bounds = textureSize(input_sampler, 0) - ivec2(1);
  value = texelFetch(input_sampler, min(xy, texture_bounds), 0).x;
}

void load_input_vector(sampler2D input_sampler, out vec3 vector)
{
  ivec2 xy = ivec2(gl_GlobalInvocationID.xy);
  ivec2 texture_bounds = textureSize(input_sampler, 0) - ivec2(1);
  vector = texelFetch(input_sampler, min(xy, texture_bounds), 0).xyz;
}

void load_input_color(sampler2D input_sampler, out vec4 color)
{
  ivec2 xy = ivec2(gl_GlobalInvocationID.xy);
  ivec2 texture_bounds = textureSize(input_sampler, 0) - ivec2(1);
  color = texelFetch(input_sampler, min(xy, texture_bounds), 0);
}
