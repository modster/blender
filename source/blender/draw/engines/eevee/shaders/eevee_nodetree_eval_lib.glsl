
#pragma BLENDER_REQUIRE(eevee_closure_lib.glsl)

/* Globals used by closure. */
ClosureDiffuse g_diffuse_data;
ClosureReflection g_reflection_data;
ClosureRefraction g_refraction_data;
ClosureVolume g_volume_data;
ClosureEmission g_emission_data;
ClosureTransparency g_transparency_data;

/* Prototype. */
void nodetree_eval(vec3 N);

/* TODO(fclem) Replace by real nodetree.  */
void nodetree_eval(vec3 N)
{
  g_diffuse_data.color = vec3(1.0, 1.0, 1.0);
  g_diffuse_data.N = N;
  g_diffuse_data.thickness = 0.0;
  g_diffuse_data.sss_radius = vec3(0);
  g_diffuse_data.sss_id = 0u;

  g_reflection_data.color = vec3(0.2, 1.0, 0.2);
  g_reflection_data.N = N;
  g_reflection_data.roughness = 0.5;

  g_refraction_data.color = vec3(0.0, 0.0, 0.0);
  g_refraction_data.N = N;
  g_refraction_data.roughness = 0.5;

  g_volume_data.emission = vec3(1.0, 0.5, 0.0);
  g_volume_data.scattering = vec3(0.5, 1.0, 0.0);
  g_volume_data.transmittance = vec3(1.0, 0.8, 0.5);
  g_volume_data.anisotropy = 0.0;

  g_emission_data.emission = vec3(0);

  g_transparency_data.transmittance = vec3(0.0);
  g_transparency_data.holdout = 0.0;
}
