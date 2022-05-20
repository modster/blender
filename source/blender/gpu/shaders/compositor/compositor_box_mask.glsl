#pragma BLENDER_REQUIRE(gpu_shader_compositor_texture_utilities.glsl)

void main()
{
  ivec2 xy = ivec2(gl_GlobalInvocationID.xy);

  vec2 uv = vec2(xy) / vec2(domain_size - ivec2(1));
  uv -= location;
  uv.y *= float(domain_size.y) / float(domain_size.x);
  uv = mat2(cos_angle, -sin_angle, sin_angle, cos_angle) * uv;
  bool is_inside = all(lessThan(abs(uv), size));

  float base_mask_value = texture_load(base_mask, xy).x;
  float value = texture_load(mask_value, xy).x;

#if defined(CMP_NODE_MASKTYPE_ADD)
  float output_mask_value = is_inside ? max(base_mask_value, value) : base_mask_value;
#elif defined(CMP_NODE_MASKTYPE_SUBTRACT)
  float output_mask_value = is_inside ? clamp(base_mask_value - value, 0.0, 1.0) : base_mask_value;
#elif defined(CMP_NODE_MASKTYPE_MULTIPLY)
  float output_mask_value = is_inside ? base_mask_value * value : 0.0;
#elif defined(CMP_NODE_MASKTYPE_NOT)
  float output_mask_value = is_inside ? (base_mask_value > 0.0 ? 0.0 : value) : base_mask_value;
#endif

  imageStore(output_mask, xy, vec4(output_mask_value));
}
