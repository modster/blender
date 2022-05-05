#pragma BLENDER_REQUIRE(gpu_shader_compositor_texture_utilities.glsl)

void main()
{
  ivec2 xy = ivec2(gl_GlobalInvocationID.xy);

  const int weights_size = texture_size(weights);
  const int blur_size = weights_size / 2;

  vec4 color = vec4(0.0);
  for (int i = 0; i < weights_size; i++) {
#if defined(BLUR_HORIZONTAL)
    const ivec2 offset = ivec2(i - blur_size, 0);
#elif defined(BLUR_VERTICAL)
    const ivec2 offset = ivec2(0, i - blur_size);
#endif
    color += texture_load(input_image, xy + offset) * texture_load(weights, i).x;
  }
  imageStore(output_image, xy, color);
}
