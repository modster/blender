#pragma BLENDER_REQUIRE(gpu_shader_compositor_texture_utils.glsl)

void main()
{
  ivec2 xy = ivec2(gl_GlobalInvocationID.xy);
  vec4 texel = texture_load(input_sampler, xy);
  imageStore(output_image, xy, CONVERT_EXPRESSION);
}
