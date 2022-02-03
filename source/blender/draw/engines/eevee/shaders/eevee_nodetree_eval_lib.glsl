
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
  float transmit_rand;
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
  float transmit_total = g_diffuse_data.color.r + g_refraction_data.color.r;
  if (g_data.transmit_rand >= 0.0) {
    float transmit_threshold = g_diffuse_data.color.r * safe_rcp(transmit_total);
    if (g_data.transmit_rand > transmit_threshold) {
      /* Signal that we will use the transmition closure. */
      g_data.transmit_rand = 0.0;
    }
    else {
      /* Signal that we will use the diffuse closure. */
      g_data.transmit_rand = 1.0;
    }
  }

  closure_weight_randomize(g_diffuse_data, g_data.closure_rand);
  closure_weight_randomize(g_reflection_data, g_data.closure_rand);
  closure_weight_randomize(g_refraction_data, g_data.closure_rand);

  /* Amend total weight to avoid loosing energy. */
  if (g_data.transmit_rand > 0.0) {
    g_diffuse_data.color.r += g_refraction_data.color.r;
  }
  else if (g_data.transmit_rand == 0.0) {
    g_refraction_data.color.r += g_diffuse_data.color.r;
  }
}

/* Prototypes. */
void attrib_load();
vec3 nodetree_displacement();
Closure nodetree_surface();
Closure nodetree_volume();
float nodetree_thickness();

#ifdef EEVEE_MATERIAL_STUBS
#define attrib_load()
#define nodetree_displacement() vec3(0)
#define nodetree_surface() CLOSURE_DEFAULT
#define nodetree_volume() CLOSURE_DEFAULT
#define nodetree_thickness() 0.1
#endif
