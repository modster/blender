
#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_object_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_gbuffer_lib.glsl)

layout(location = 0) out uvec4 out_diffuse_data;     /* Diffuse BSDF, BSSSDF, Translucency. */
layout(location = 1) out uvec2 out_reflection_data;  /* Glossy BSDF. */
layout(location = 2) out uvec4 out_refraction_data;  /* Refraction BSDF. */
layout(location = 3) out uvec4 out_volume_data;      /* Volume Emission, Absorption, Scatter. */
layout(location = 4) out vec3 out_emission_data;     /* Emission. */
layout(location = 5) out vec4 out_transparency_data; /* Transparent BSDF, Holdout. */

MeshData g_surf;

void main(void)
{
  g_surf = init_from_interp();

  GBufferDiffuseData diff;
  diff.color = vec3(1.0, 1.0, 1.0);
  diff.N = g_surf.N;
  diff.thickness = 0.0;
  diff.sss_radius = vec3(0);
  diff.sss_id = 0u;

  GBufferReflectionData refl;
  refl.color = vec3(0.2, 1.0, 0.2);
  refl.N = g_surf.N;
  refl.roughness = 0.5;

  out_diffuse_data = gbuffer_encode_diffuse_data(diff);
  out_reflection_data = gbuffer_encode_reflection_data(refl);
}