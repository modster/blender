
/**
 * Apply heterogeneous volume lighting and evaluates homogeneous volumetrics if needed.
 *
 * We read volume parameters from the gbuffer and consider them constant for the whole volume.
 * This only applies to solid objects not volumes.
 **/

#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_bsdf_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_culling_iter_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shadow_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_gbuffer_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_volume_eval_lib.glsl)

void main(void)
{
  ClosureVolume volume_data = gbuffer_load_volume_data(volume_data_tx, uvcoordsvar.xy);

  /* For volumes from solid objects. */
  // float depth_max = linear_z(texture(depth_max_tx, uv).r);
  // float depth_min = linear_z(texture(depth_min_tx, uv).r);

  /* Refine bounds to skip empty areas. */
  // float dist_from_bbox = intersect_bbox_ray(P, V, bbox);
  // depth_min = max(dist_from_bbox, depth_min);

  vec3 volume_radiance;
  if (volume_data.anisotropy == VOLUME_HETEROGENEOUS) {
    volume_radiance = volume_data.scattering;
  }
  else {
    // volume_eval_homogeneous(P, depth_min, depth_max, volume_radiance);
    volume_radiance = vec3(0.0);
  }

  /* Apply transmittance of surface on volumetric radiance because
   * the volume is behind the surface.  */
  ClosureTransparency transparency_data = gbuffer_load_transparency_data(transparency_data_tx,
                                                                         uvcoordsvar.xy);
  volume_radiance *= transparency_data.transmittance;

  out_combined = vec4(volume_radiance, 0.0);
  out_volume = volume_radiance;
}
