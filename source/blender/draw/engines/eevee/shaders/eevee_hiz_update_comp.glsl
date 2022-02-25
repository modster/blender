/**
 * Shader that down-sample depth buffer, creating a Hierarchical-Z buffer.
 * Saves max value of each 2x2 texel in the mipmap above the one we are rendering to.
 * Adapted from http://rastergrid.com/blog/2010/10/hierarchical-z-map-based-occlusion-culling/
 *
 * Major simplification has been made since we pad the buffer to always be bigger than input to
 * avoid mipmapping misalignement.
 *
 * Start by copying the base level by quad loading the depth.
 * Then each thread compute it's local depth for level 1.
 * After that we use shared variables to do inter thread comunication and downsample to max level.
 */

#pragma BLENDER_REQUIRE(common_math_lib.glsl)

shared float local_depths[16][16];

/* Load values from the previous lod level. */
vec4 load_local_depths(uvec2 pixel)
{
  pixel *= 2u;
  return vec4(local_depths[pixel.x + 0u][pixel.y + 1u],
              local_depths[pixel.x + 1u][pixel.y + 1u],
              local_depths[pixel.x + 1u][pixel.y + 0u],
              local_depths[pixel.x + 0u][pixel.y + 0u]);
}

void store_local_depth(uvec2 pixel, float depth)
{
  local_depths[pixel.x][pixel.y] = depth;
}

void main()
{
  uvec2 local_px = gl_LocalInvocationID.xy;
  ivec2 kernel_origin = ivec2(gl_WorkGroupSize.xy * gl_WorkGroupID.xy);

  /* Copy level 0. */
  ivec2 dst_px = ivec2(kernel_origin + local_px) * 2;
  vec2 samp_co = (vec2(dst_px) + 0.5) / vec2(textureSize(depth_tx, 0));
  vec4 samp = textureGather(depth_tx, samp_co);
  imageStore(out_lvl0, dst_px + ivec2(0, 1), samp.xxxx);
  imageStore(out_lvl0, dst_px + ivec2(1, 1), samp.yyyy);
  imageStore(out_lvl0, dst_px + ivec2(1, 0), samp.zzzz);
  imageStore(out_lvl0, dst_px + ivec2(0, 0), samp.wwww);

  /* Level 1. (No load) */
  float max_depth = max_v4(samp);
  dst_px = ivec2(kernel_origin + local_px);
  imageStore(out_lvl1, dst_px, vec4(max_depth));
  store_local_depth(local_px, max_depth);

  /* This is a define because compilers have different requirements about image qualifiers in
   * function arguments. */
#define downsample_level(out_lvl_, lod_) \
  barrier(); \
  if (all(lessThan(local_px, uvec2(gl_WorkGroupSize.xy >> (lod_ - 1u))))) { \
    samp = load_local_depths(local_px); \
    max_depth = max_v4(samp); \
    dst_px = ivec2((kernel_origin >> (lod_ - 1u)) + local_px); \
    imageStore(out_lvl_, dst_px, vec4(max_depth)); \
    store_local_depth(local_px, max_depth); \
  }

  /* Level 2-5. */
  downsample_level(out_lvl2, 2u);
  downsample_level(out_lvl3, 3u);
  downsample_level(out_lvl4, 4u);
  downsample_level(out_lvl5, 5u);
}
