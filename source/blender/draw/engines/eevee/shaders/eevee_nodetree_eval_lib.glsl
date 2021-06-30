
#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_closure_lib.glsl)
#pragma BLENDER_REQUIRE(gpu_shader_codegen_lib.glsl)

/* Globals used by closure. */
ClosureDiffuse g_diffuse_data;
ClosureReflection g_reflection_data;
ClosureRefraction g_refraction_data;
ClosureVolume g_volume_data;
ClosureEmission g_emission_data;
ClosureTransparency g_transparency_data;

struct GlobalData {
  /** World position. */
  vec3 P;
  /** Surface Normal. */
  vec3 N;
  /** Geometric Normal. */
  vec3 Ng;
  /** Barycentric coordinates. */
  vec2 barycentric_coords;
  vec3 barycentric_dists;
  /** Ray properties (approximation). */
  int ray_type;
  float ray_depth;
  float ray_length;
  /** Random number to sample a closure. */
  float closure_rand;
  /** Hair time along hair length. 0 at base 1 at tip. */
  float hair_time;
  /** Hair time along width of the hair. */
  float hair_time_width;
  /** Hair thickness in world space. */
  float hair_thickness;
  /** Index of the strand for per strand effects. */
  int hair_strand_id;
  /** Is hair. */
  bool is_strand;
};

GlobalData g_data;

void ntree_eval_init()
{
  g_diffuse_data.color = vec3(0.0);
  g_diffuse_data.N = vec3(1.0, 0.0, 0.0);
  g_diffuse_data.thickness = 0.0;
  g_diffuse_data.sss_radius = vec3(0);
  g_diffuse_data.sss_id = 0u;

  g_reflection_data.color = vec3(0.0);
  g_reflection_data.N = vec3(1.0, 0.0, 0.0);
  g_reflection_data.roughness = 0.5;

  g_refraction_data.color = vec3(0.0);
  g_refraction_data.N = vec3(1.0, 0.0, 0.0);
  g_refraction_data.roughness = 0.5;

  g_volume_data.emission = vec3(0.0);
  g_volume_data.scattering = vec3(0.0);
  g_volume_data.transmittance = vec3(1.0);
  g_volume_data.anisotropy = 0.0;

  g_emission_data.emission = vec3(0.0);

  g_transparency_data.transmittance = vec3(0.0);
  g_transparency_data.holdout = 0.0;
}

void ntree_eval_weights()
{
  closure_weight_randomize(g_diffuse_data, g_data.closure_rand);
  closure_weight_randomize(g_reflection_data, g_data.closure_rand);
  closure_weight_randomize(g_refraction_data, g_data.closure_rand);
}
