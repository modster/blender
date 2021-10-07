
/**
 * Postprocess diffuse radiance output from the diffuse evaluation pass to mimic subsurface
 * transmission.
 *
 * This implementation follows the technique described in the siggraph presentation:
 * "Efficient screen space subsurface scattering Siggraph 2018"
 * by Evgenii Golubev
 *
 * But, instead of having all the precomputed weights for all three color primaries,
 * we precompute a weight profile texture to be able to support per pixel AND per channel radius.
 **/

#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(common_math_geom_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_sampling_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_gbuffer_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_closure_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shader_shared.hh)

layout(std140) uniform subsurface_block
{
  SubsurfaceData sss;
};

uniform sampler2D depth_tx;
uniform sampler2D radiance_tx;
uniform sampler2D transmit_color_tx;
uniform sampler2D transmit_normal_tx;
uniform sampler2D transmit_data_tx;

in vec4 uvcoordsvar;

layout(location = 0) out vec4 out_combined;
/* TODO(fclem) Output to diffuse pass without feedback loop. */

vec3 burley_setup(vec3 radius, vec3 albedo)
{
  /* Scale albedo because we can have HDR value caused by BSDF sampling. */
  vec3 A = albedo / max(1e-6, max_v3(albedo));
  /* Diffuse surface transmission, equation (6). */
  vec3 s = 1.9 - A + 3.5 * sqr(A - 0.8);
  /* Mean free path length adapted to fit ancient Cubic and Gaussian models. */
  vec3 l = 0.25 * M_1_PI * radius;

  return l / s;
}

vec3 burley_eval(vec3 d, float r)
{
  /* Slide 33. */
  vec3 exp_r_3_d = exp(-r / (3.0 * d));
  vec3 exp_r_d = exp_r_3_d * exp_r_3_d * exp_r_3_d;
  /** NOTE:
   * - Surface albedo is applied at the end.
   * - This is normalized diffuse model, so the equation is multiplied
   *   by 2*pi, which also matches cdf().
   */
  return (exp_r_d + exp_r_3_d) / (4.0 * d);
}

void main(void)
{
  vec2 center_uv = uvcoordsvar.xy;

  vec3 vP = get_view_space_from_depth(center_uv, texture(depth_tx, center_uv).r);
  vec4 tra_col_in = texture(transmit_color_tx, center_uv);
  vec4 tra_nor_in = texture(transmit_normal_tx, center_uv);
  vec4 tra_dat_in = texture(transmit_data_tx, center_uv);

  if (tra_nor_in.x < 0.0) {
    /* Refraction transmission case. */
    out_combined = vec4(0.0);
    return;
  }

  ClosureDiffuse diffuse = gbuffer_load_diffuse_data(tra_col_in, tra_nor_in, tra_dat_in);

  float max_radius = max_v3(diffuse.sss_radius);

  float homcoord = ProjectionMatrix[2][3] * vP.z + ProjectionMatrix[3][3];
  vec2 sample_scale = vec2(ProjectionMatrix[0][0], ProjectionMatrix[1][1]) *
                      (0.5 * max_radius / homcoord);

  float pixel_footprint = sample_scale.x * float(textureSize(radiance_tx, 0).x);
  if (pixel_footprint <= 1.0) {
    /* Early out. */
    out_combined = vec4(texture(radiance_tx, center_uv).rgb * diffuse.color, 0.0);
    return;
  }

  diffuse.sss_radius = max(vec3(1e-4), diffuse.sss_radius / max_radius) * max_radius;
  vec3 d = burley_setup(diffuse.sss_radius, diffuse.color);

  /* Do not rotate too much to avoid too much cache misses. */
  float golden_angle = M_PI * (3.0 - sqrt(5.0));
  float theta = interlieved_gradient_noise(gl_FragCoord.xy, 0, 0.0) * golden_angle;
  float cos_theta = cos(theta);
  float sin_theta = sqrt(1.0 - sqr(cos_theta));
  mat2 rot = mat2(cos_theta, sin_theta, -sin_theta, cos_theta);

  mat2 scale = mat2(sample_scale.x, 0.0, 0.0, sample_scale.y);
  mat2 sample_space = scale * rot;

  vec3 accum_weight = vec3(0.0);
  vec3 accum = vec3(0.0);

  /* TODO/OPTI(fclem) Make separate sample set for lower radius. */

  for (int i = 0; i < sss.sample_len; i++) {
    vec2 sample_uv = center_uv + sample_space * sss.samples[i].xy;
    float pdf = sss.samples[i].z;

    float sample_depth = texture(depth_tx, sample_uv).r;
    vec3 sample_vP = get_view_space_from_depth(sample_uv, sample_depth);

    vec4 sample_data = texture(radiance_tx, sample_uv);
    vec3 sample_radiance = sample_data.rgb;
    uint sample_sss_id = uint(sample_data.a * 1024.0);

    if (sample_sss_id != diffuse.sss_id) {
      continue;
    }

    /* Discard out of bounds samples. */
    if (any(lessThan(sample_uv, vec2(0.0))) || any(greaterThan(sample_uv, vec2(1.0)))) {
      continue;
    }

    /* Slide 34. */
    float r = distance(sample_vP, vP);
    vec3 weight = burley_eval(d, r) / pdf;

    accum += sample_radiance * weight;
    accum_weight += weight;
  }
  /* Normalize the sum (slide 34). */
  accum /= accum_weight;
  /* Apply surface color on final radiance. */
  accum *= diffuse.color;

  /* Debug, detect NaNs. */
  if (any(isnan(accum))) {
    accum = vec3(1.0, 0.0, 1.0);
  }

  out_combined = vec4(accum, 0.0);
}
