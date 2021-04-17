
/**
 * Debug Shader outputing a gradient of orange - white - blue to mark culling hotspots.
 * Green pixels are error pixels that are missing lights from the culling pass (i.e: when culling
 * pass is not conservative enough). This shader will only work on the last light batch so remove
 * some lights from the scene you are debugging to have below CULLING_ITEM_BATCH lights.
 */

#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_culling_iter_lib.glsl)

layout(std140) uniform lights_block
{
  LightData lights[CULLING_ITEM_BATCH];
};

layout(std140) uniform lights_culling_block
{
  CullingData culling;
};

uniform usampler2D item_culling_tx;
uniform sampler2D depth_tx;

in vec4 uvcoordsvar;

layout(location = 0) out vec4 out_debug_color;

void main(void)
{
  float depth = textureLod(depth_tx, uvcoordsvar.xy, 0.0).r;
  float vP_z = get_view_z_from_depth(depth);

  vec3 P = get_world_space_from_depth(uvcoordsvar.xy, depth);

  float lights_count = 0.0;
  uint lights_cull = 0u;
  ITEM_FOREACH_BEGIN (culling, item_culling_tx, vP_z, l_idx) {
    LightData light = lights[l_idx];
    lights_cull |= 1u << l_idx;
    lights_count += 1.0;
  }
  ITEM_FOREACH_END

  uint lights_nocull = 0u;
  ITEM_FOREACH_BEGIN_NO_CULL (culling, l_idx) {
    LightData light = lights[l_idx];
    if (distance(light._position, P) < light.influence_radius_max) {
      lights_nocull |= 1u << l_idx;
    }
  }
  ITEM_FOREACH_END

  if ((lights_cull & lights_nocull) != lights_nocull) {
    /* ERROR. Some lights were culled incorrectly. */
    out_debug_color = vec4(0.0, 1.0, 0.0, 1.0);
  }
  else {
    out_debug_color = vec4(heatmap_gradient(lights_count / 16.0), 1.0);
  }
}