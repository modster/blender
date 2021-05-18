
#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_surface_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_gbuffer_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_nodetree_eval_lib.glsl)

layout(location = 0) out uvec4 out_diffuse_data;     /* Diffuse BSDF, BSSSDF, Translucency. */
layout(location = 1) out uvec2 out_reflection_data;  /* Glossy BSDF. */
layout(location = 2) out uvec4 out_refraction_data;  /* Refraction BSDF. */
layout(location = 3) out uvec4 out_volume_data;      /* Volume Emission, Absorption, Scatter. */
layout(location = 4) out vec4 out_emission_data;     /* Emission. */
layout(location = 5) out vec4 out_transparency_data; /* Transparent BSDF, Holdout. */

GlobalData g_data;

void main(void)
{
  g_data = init_from_interp();

  nodetree_eval(g_data.N);

  out_diffuse_data = gbuffer_store_diffuse_data(g_diffuse_data);
  out_reflection_data = gbuffer_store_reflection_data(g_reflection_data);
  out_refraction_data = gbuffer_store_refraction_data(g_refraction_data);
  out_volume_data = gbuffer_store_volume_data(g_volume_data);
  out_emission_data = gbuffer_store_emission_data(g_emission_data);
  out_transparency_data = gbuffer_store_transparency_data(g_transparency_data);
}
