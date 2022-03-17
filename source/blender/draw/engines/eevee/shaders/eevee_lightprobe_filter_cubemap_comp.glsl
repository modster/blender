
/**
 * Simple single pass downsampling to fill a mipmap pyramid.
 */

#pragma BLENDER_REQUIRE(eevee_sampling_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_bsdf_sampling_lib.glsl)

shared vec4 shared_values[16][16];

vec4 brightness_clamp(vec4 color)
{
  float luma = max_v3(color.rgb);
  color.rgb *= 1.0 - max(0.0, luma - filter_buf.luma_max) * safe_rcp(luma);
  return color;
}

void main()
{
  uvec2 local_px = gl_LocalInvocationID.xy;
  ivec2 kernel_origin = ivec2(gl_WorkGroupSize.xy * gl_WorkGroupID.xy);

  int layer = int(gl_GlobalInvocationID.z);

  /* Copy level 0. */
  ivec2 dst_px = ivec2(kernel_origin + local_px) * 2;
  vec4 samp0 = texelFetch(radiance_tx, ivec3(dst_px + ivec2(0, 1), layer), 0);
  vec4 samp1 = texelFetch(radiance_tx, ivec3(dst_px + ivec2(1, 1), layer), 0);
  vec4 samp2 = texelFetch(radiance_tx, ivec3(dst_px + ivec2(1, 0), layer), 0);
  vec4 samp3 = texelFetch(radiance_tx, ivec3(dst_px + ivec2(0, 0), layer), 0);
  vec4 dst_color;

  /* Apply clamping on the source to avoid propagating light in lower mips. */
  samp0 = brightness_clamp(samp0);
  samp1 = brightness_clamp(samp1);
  samp2 = brightness_clamp(samp2);
  samp3 = brightness_clamp(samp3);

  if (!filter_buf.skip_mip_0) {
    imageStore(out_lvl0, ivec3(dst_px + ivec2(0, 1), layer), samp0);
    imageStore(out_lvl0, ivec3(dst_px + ivec2(1, 1), layer), samp1);
    imageStore(out_lvl0, ivec3(dst_px + ivec2(1, 0), layer), samp2);
    imageStore(out_lvl0, ivec3(dst_px + ivec2(0, 0), layer), samp3);
  }

  if (1u <= filter_buf.out_lod_max) {
    /* Level 1. (No load) */
    dst_color = avg4(samp0, samp1, samp2, samp3);
    dst_px = ivec2(kernel_origin + local_px);
    imageStore(out_lvl1, ivec3(dst_px, layer), dst_color);
    shared_values[local_px.x][local_px.y] = dst_color;
  }

  /* This is a define because compilers have different requirements about image qualifiers in
   * function arguments. */
#define downsample_level(out_lvl_, lod_) \
  barrier(); \
  if (lod_ <= filter_buf.out_lod_max && \
      all(lessThan(local_px, uvec2(gl_WorkGroupSize.xy >> (lod_ - 1u))))) { \
    uvec2 shared_px = local_px * 2u; \
    /* Load values from the previous lod level. */ \
    samp0 = shared_values[shared_px.x + 0u][shared_px.y + 1u]; \
    samp1 = shared_values[shared_px.x + 1u][shared_px.y + 1u]; \
    samp2 = shared_values[shared_px.x + 1u][shared_px.y + 0u]; \
    samp3 = shared_values[shared_px.x + 0u][shared_px.y + 0u]; \
    dst_color = avg4(samp0, samp1, samp2, samp3); \
    dst_px = ivec2((kernel_origin >> (lod_ - 1u)) + local_px); \
    imageStore(out_lvl_, ivec3(dst_px, layer), dst_color); \
  } \
  barrier(); \
  if (lod_ <= filter_buf.out_lod_max && \
      all(lessThan(local_px, uvec2(gl_WorkGroupSize.xy >> (lod_ - 1u))))) { \
    shared_values[local_px.x][local_px.y] = dst_color; \
  }

  /* Level 2-5. */
  downsample_level(out_lvl2, 2u);
  downsample_level(out_lvl3, 3u);
  downsample_level(out_lvl4, 4u);
  downsample_level(out_lvl5, 5u);

  /* TODO(fclem): Support longer mipchain. */
}
