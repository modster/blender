
/**
 * Forward lighting evaluation: Lighting is evaluated during the geometry rasterization.
 *
 * This is used by alpha blended materials and materials using Shader to RGB nodes.
 **/

#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_nodetree_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_surf_lib.glsl)

void main(void)
{
  init_globals();

  fragment_displacement();

  nodetree_surface();

  g_holdout = saturate(g_holdout);

  out_radiance.rgb = g_emission;
  out_radiance.rgb += g_diffuse_data.color * saturate(g_diffuse_data.N.z);
  out_radiance.rgb += g_reflection_data.color * saturate(g_reflection_data.N.z);
  out_radiance.a = 0.0;

  out_radiance.rgb *= 1.0 - g_holdout;

  out_transmittance.rgb = g_transmittance;
  out_transmittance.a = saturate(avg(g_transmittance));

  /* Test */
  out_transmittance.a = 1.0 - out_transmittance.a;
  out_radiance.a = 1.0 - out_radiance.a;
}
