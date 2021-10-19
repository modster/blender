/**
 * Shader that down-sample depth buffer, creating a Hierarchical-Z buffer.
 * Saves max value of each 2x2 texel in the mipmap above the one we are rendering to.
 * Adapted from http://rastergrid.com/blog/2010/10/hierarchical-z-map-based-occlusion-culling/
 *
 * Major simplification has been made since we pad the buffer to always be bigger than input to
 * avoid mipmapping misalignement.
 */

#pragma BLENDER_REQUIRE(common_math_lib.glsl)

uniform sampler2D depth_tx;

uniform vec2 texel_size;

layout(location = 0) out float out_depth;

#ifndef GPU_ARB_texture_gather
vec4 texGather(sampler2D tex, vec2 uv)
{
  vec4 ofs = vec2(0.5, 0.5, -0.5, -0.5) * texel_size.xyxy;
  return vec4(texture(tex, uv + ofs.zw).r,
              texture(tex, uv + ofs.zy).r,
              texture(tex, uv + ofs.xw).r,
              texture(tex, uv + ofs.xy).r);
}
#else
#  define texGather(a, b) textureGather(a, b)
#endif

void main()
{
  /* NOTE(@fclem): textureSize() does not work the same on all implementations
   * when changing the min and max texture levels. Use uniform instead (see T87801). */
  vec2 uv = gl_FragCoord.xy * texel_size;

  vec4 samp = texGather(depth_tx, uv);

  out_depth = max_v4(samp);
}
