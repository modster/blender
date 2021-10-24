
/**
 * Outputs pre-convolved diffuse irradiance from an input radiance cubemap.
 * Input radiance has its mipmaps updated so we can use filtered importance sampling.
 * The output is a really small 6 directional basis similar to the ambient cube technique.
 * Downside: very very low resolution (6 texels), bleed lighting because of interpolation.
 *
 * https://cdn.cloudflare.steamstatic.com/apps/valve/2006/SIGGRAPH06_Course_ShadingInValvesSourceEngine.pdf
 */

#pragma BLENDER_REQUIRE(eevee_irradiance_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_sampling_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_bsdf_sampling_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_lightprobe_filter_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shader_shared.hh)

uniform samplerCube radiance_tx;

layout(std140) uniform filter_block
{
  LightProbeFilterData probe;
};

layout(location = 0) out vec4 out_irradiance;

void main()
{
  int x = int(gl_FragCoord.x) % 3;
  int y = int(gl_FragCoord.y) % 2;

  vec3 N, T, B;
  switch (x) {
    default:
      N = vec3(1.0, 0.0, 0.0);
      break;
    case 1:
      N = vec3(0.0, 1.0, 0.0);
      break;
    case 2:
      N = vec3(0.0, 0.0, 1.0);
      break;
  }

  if (y == 1) {
    N = -N;
  }

  make_orthonormal_basis(N, T, B);

  /* Integrating Envmap. */
  float weight = 0.0;
  vec3 radiance = vec3(0.0);
  for (float i = 0; i < probe.sample_count; i++) {
    vec3 Xi = sample_cylinder(hammersley_2d(i, probe.sample_count));

    float pdf;
    vec3 L = sample_uniform_hemisphere(Xi, N, T, B, pdf);
    float NL = dot(N, L);

    if (NL > 0.0) {
      /* Coarse Approximation of the mapping distortion.
       * Unit Sphere -> Cubemap Face. */
      const float dist = 4.0 * M_PI / 6.0;
      /* http://http.developer.nvidia.com/GPUGems3/gpugems3_ch20.html : Equation 13 */
      float lod = probe.lod_bias - 0.5 * log2(pdf * dist);

      radiance += textureLod(radiance_tx, L, lod).rgb * NL;
      weight += NL;
    }
  }

  out_irradiance = irradiance_encode(probe.instensity_fac * radiance / weight);
}
