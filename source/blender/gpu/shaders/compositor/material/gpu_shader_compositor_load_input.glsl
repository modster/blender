void load_input_float(sampler2D input_sampler, out float value)
{
  ivec2 xy = ivec2(gl_GlobalInvocationID.xy);
  value = texture_load(input_sampler, xy).x;
}

void load_input_vector(sampler2D input_sampler, out vec3 vector)
{
  ivec2 xy = ivec2(gl_GlobalInvocationID.xy);
  vector = texture_load(input_sampler, xy).xyz;
}

void load_input_color(sampler2D input_sampler, out vec4 color)
{
  ivec2 xy = ivec2(gl_GlobalInvocationID.xy);
  color = texture_load(input_sampler, xy);
}
