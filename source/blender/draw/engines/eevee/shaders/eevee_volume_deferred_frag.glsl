
/**
 * Renders heterogeneous volumes.
 **/

#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_volume_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_volume_eval_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_gbuffer_lib.glsl)

uniform sampler2D depth_max_tx;

layout(location = 0) out uvec4 out_volume_data;      /* Volume Emission, Absorption, Scatter. */
layout(location = 1) out vec4 out_transparency_data; /* Transparent BSDF, Holdout. */

void main(void)
{
  // g_volume = init_from_interp();

  vec2 uv = gl_FragCoord.xy / vec2(textureSize(depth_max_tx, 0).xy);

  /* For volumes from solid objects. */
  vec3 vP_start = get_view_space_from_depth(uv, gl_FragCoord.z);
  vec3 vP_end = get_view_space_from_depth(uv, texture(depth_max_tx, uv).r);

  Ray ray;
  ray.origin = vP_start;
  ray.direction = vP_end - vP_start;
  ray.direction /= abs(ray.direction.z);
  ray.max_time = max(vP_start.z - vP_end.z, 0.0);

  /* Refine bounds to skip empty areas. */
  // float dist = line_unit_box_intersect_dist(ls_ray_ori, ls_ray_dir);
  // if (dist > 0.0) {
  //   ls_ray_ori = ls_ray_dir * dist + ls_ray_ori;
  // }

  // vec3 ls_vol_isect = ls_ray_end - ls_ray_ori;
  // if (dot(ls_ray_dir, ls_vol_isect) < 0.0) {
  //   /* Start is further away than the end.
  //    * That means no volume is intersected. */
  //   discard;
  // }

  vec3 out_depth_time;
  vec3 out_radiance = vec3(0.0);
  vec3 out_transmittance = vec3(1.0);
  // volume_eval_scattering_transmittance(
  //     P, depth_min, depth_max, out_radiance, out_transmittance, gl_FragDepth);

  volume_eval_homogenous(ray, out_transmittance, out_depth_time);

  gl_FragDepth = get_depth_from_view_z(ray.origin.z - avg(out_depth_time));

  g_volume_data.emission = vec3(0);
  g_volume_data.scattering = out_radiance;
  g_volume_data.transmittance = out_transmittance;
  g_volume_data.anisotropy = VOLUME_HETEROGENEOUS;

  g_transparency_data.transmittance = vec3(1.0);
  g_transparency_data.holdout = 0.0;

  out_volume_data = gbuffer_store_volume_data(g_volume_data);
  out_transparency_data = gbuffer_store_transparency_data(g_transparency_data);
}
