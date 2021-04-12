
#pragma BLENDER_REQUIRE(eevee_motion_blur_lib.glsl)

/* Converts uv velocity into pixel space. Assumes velocity_tx is the same resolution as the
 * target post-fx framebuffer. */
vec4 sample_velocity(MotionBlurData mb, sampler2D velocity_tx, ivec2 texel)
{
  vec4 velocity = texelFetch(velocity_tx, texel, 0);
  velocity *= vec2(textureSize(velocity_tx, 0)).xyxy;
  velocity = (mb.is_viewport) ? velocity.xyxy : velocity;
  return velocity;
}
vec2 sample_velocity(MotionBlurData mb, sampler2D velocity_tx, vec2 uv, const bool next)
{
  vec4 velocity = texture(velocity_tx, uv);
  velocity *= vec2(textureSize(velocity_tx, 0)).xyxy;
  return (next && !mb.is_viewport) ? velocity.zw : velocity.xy;
}
