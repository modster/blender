#pragma BLENDER_REQUIRE(gpu_shader_compositor_texture_utils.glsl)

void main()
{
  ivec2 xy = ivec2(gl_GlobalInvocationID.xy);
  imageStore(output_image, xy, texture_load(input_image, xy + lower_bound));
}
