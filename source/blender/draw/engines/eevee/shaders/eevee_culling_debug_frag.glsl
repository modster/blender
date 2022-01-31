
/**
 * Debug Shader outputing a gradient of orange - white - blue to mark culling hotspots.
 * Green pixels are error pixels that are missing lights from the culling pass (i.e: when culling
 * pass is not conservative enough).
 */

#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_light_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_culling_iter_lib.glsl)

layout(std430, binding = 0) readonly restrict buffer lights_buf
{
  LightData lights[];
};

layout(std430, binding = 1) readonly restrict buffer lights_zbins_buf
{
  CullingZBin lights_zbins[];
};

layout(std430, binding = 2) readonly restrict buffer lights_culling_buf
{
  CullingData light_culling;
};

layout(std430, binding = 3) readonly restrict buffer lights_tile_buf
{
  CullingWord lights_culling_words[];
};

uniform sampler2D depth_tx;

in vec4 uvcoordsvar;

layout(location = 0) out vec4 out_debug_color;

void main(void)
{
  float depth = texelFetch(depth_tx, ivec2(gl_FragCoord.xy), 0).r;
  float vP_z = get_view_z_from_depth(depth);

  vec3 P = get_world_space_from_depth(uvcoordsvar.xy, depth);

  float lights_count = 0.0;
  uint lights_cull = 0u;
  LIGHT_FOREACH_BEGIN_LOCAL (
      light_culling, lights_zbins, lights_culling_words, gl_FragCoord.xy, vP_z, l_idx) {
    LightData light = lights[l_idx];
    lights_cull |= 1u << l_idx;
    lights_count += 1.0;
  }
  LIGHT_FOREACH_END

  uint lights_nocull = 0u;
  LIGHT_FOREACH_BEGIN_LOCAL_NO_CULL (light_culling, l_idx) {
    LightData light = lights[l_idx];
    if (distance(light._position, P) < light.influence_radius_max) {
      lights_nocull |= 1u << l_idx;
    }
  }
  LIGHT_FOREACH_END

  if ((lights_cull & lights_nocull) != lights_nocull) {
    /* ERROR. Some lights were culled incorrectly. */
    out_debug_color = vec4(0.0, 1.0, 0.0, 1.0);
  }
  else {
    out_debug_color = vec4(heatmap_gradient(lights_count / 4.0), 1.0);
  }
}