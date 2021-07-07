/*
 * Copyright 2011-2021 Blender Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "kernel/integrator/integrator_intersect_closest.h"
#include "kernel/integrator/integrator_volume_stack.h"

CCL_NAMESPACE_BEGIN

#ifdef __VOLUME__

/* Events for probalistic scattering */

typedef enum VolumeIntegrateEvent {
  VOLUME_PATH_SCATTERED = 0,
  VOLUME_PATH_ATTENUATED = 1,
  VOLUME_PATH_MISSED = 2
} VolumeIntegrateEvent;

typedef struct VolumeIntegrateResult {
  /* Throughput and offset for direct light scattering. */
  VolumeSampleMethod direct_sample_method;
  bool direct_scatter;
  float3 direct_throughput;
  float direct_t;
  ShaderVolumePhases direct_phases;

  /* Throughput and offset for indirect light scattering. */
  bool indirect_scatter;
  float3 indirect_throughput;
  float indirect_t;
  ShaderVolumePhases indirect_phases;
} VolumeIntegrateResult;

/* Ignore paths that have volume throughput below this value, to avoid unnecessary work
 * and precision issues.
 * todo: this value could be tweaked or turned into a probability to avoid unnecessary
 * work in volumes and subsurface scattering. */
#  define VOLUME_THROUGHPUT_EPSILON 1e-6f

/* Volume shader properties
 *
 * extinction coefficient = absorption coefficient + scattering coefficient
 * sigma_t = sigma_a + sigma_s */

typedef struct VolumeShaderCoefficients {
  float3 sigma_t;
  float3 sigma_s;
  float3 emission;
} VolumeShaderCoefficients;

/* Evaluate shader to get extinction coefficient at P. */
ccl_device_inline bool shadow_volume_shader_sample(INTEGRATOR_STATE_ARGS,
                                                   ShaderData *ccl_restrict sd,
                                                   float3 *ccl_restrict extinction)
{
  shader_eval_volume(INTEGRATOR_STATE_PASS, sd, PATH_RAY_SHADOW, [=](const int i) {
    return integrator_state_read_shadow_volume_stack(INTEGRATOR_STATE_PASS, i);
  });

  if (!(sd->flag & SD_EXTINCTION)) {
    return false;
  }

  const float density = object_volume_density(kg, sd->object);
  *extinction = sd->closure_transparent_extinction * density;
  return true;
}

/* Evaluate shader to get absorption, scattering and emission at P. */
ccl_device_inline bool volume_shader_sample(INTEGRATOR_STATE_ARGS,
                                            ShaderData *ccl_restrict sd,
                                            VolumeShaderCoefficients *coeff)
{
  const int path_flag = INTEGRATOR_STATE(path, flag);
  shader_eval_volume(INTEGRATOR_STATE_PASS, sd, path_flag, [=](const int i) {
    return integrator_state_read_volume_stack(INTEGRATOR_STATE_PASS, i);
  });

  if (!(sd->flag & (SD_EXTINCTION | SD_SCATTER | SD_EMISSION))) {
    return false;
  }

  coeff->sigma_s = zero_float3();
  coeff->sigma_t = (sd->flag & SD_EXTINCTION) ? sd->closure_transparent_extinction : zero_float3();
  coeff->emission = (sd->flag & SD_EMISSION) ? sd->closure_emission_background : zero_float3();

  if (sd->flag & SD_SCATTER) {
    for (int i = 0; i < sd->num_closure; i++) {
      const ShaderClosure *sc = &sd->closure[i];

      if (CLOSURE_IS_VOLUME(sc->type)) {
        coeff->sigma_s += sc->weight;
      }
    }
  }

  const float density = object_volume_density(kg, sd->object);
  coeff->sigma_s *= density;
  coeff->sigma_t *= density;
  coeff->emission *= density;

  return true;
}

ccl_device_forceinline void volume_step_init(const KernelGlobals *kg,
                                             const RNGState *rng_state,
                                             const float object_step_size,
                                             float t,
                                             float *step_size,
                                             float *step_shade_offset,
                                             float *steps_offset,
                                             int *max_steps)
{
  if (object_step_size == FLT_MAX) {
    /* Homogeneous volume. */
    *step_size = t;
    *step_shade_offset = 0.0f;
    *steps_offset = 1.0f;
    *max_steps = 1;
  }
  else {
    /* Heterogeneous volume. */
    *max_steps = kernel_data.integrator.volume_max_steps;
    float step = min(object_step_size, t);

    /* compute exact steps in advance for malloc */
    if (t > *max_steps * step) {
      step = t / (float)*max_steps;
    }

    *step_size = step;

    /* Perform shading at this offset within a step, to integrate over
     * over the entire step segment. */
    *step_shade_offset = path_state_rng_1D_hash(kg, rng_state, 0x1e31d8a4);

    /* Shift starting point of all segment by this random amount to avoid
     * banding artifacts from the volume bounding shape. */
    *steps_offset = path_state_rng_1D_hash(kg, rng_state, 0x3d22c7b3);
  }
}

/* Volume Shadows
 *
 * These functions are used to attenuate shadow rays to lights. Both absorption
 * and scattering will block light, represented by the extinction coefficient. */

#  if 0
/* homogeneous volume: assume shader evaluation at the starts gives
 * the extinction coefficient for the entire line segment */
ccl_device void volume_shadow_homogeneous(INTEGRATOR_STATE_ARGS,
                                          Ray *ccl_restrict ray,
                                          ShaderData *ccl_restrict sd,
                                          float3 *ccl_restrict throughput)
{
  float3 sigma_t = zero_float3();

  if (shadow_volume_shader_sample(INTEGRATOR_STATE_PASS, sd, &sigma_t)) {
    *throughput *= volume_color_transmittance(sigma_t, ray->t);
  }
}
#  endif

/* heterogeneous volume: integrate stepping through the volume until we
 * reach the end, get absorbed entirely, or run out of iterations */
ccl_device void volume_shadow_heterogeneous(INTEGRATOR_STATE_ARGS,
                                            Ray *ccl_restrict ray,
                                            ShaderData *ccl_restrict sd,
                                            float3 *ccl_restrict throughput,
                                            const float object_step_size)
{
  /* Load random number state. */
  RNGState rng_state;
  shadow_path_state_rng_load(INTEGRATOR_STATE_PASS, &rng_state);

  float3 tp = *throughput;

  /* Prepare for stepping.
   * For shadows we do not offset all segments, since the starting point is
   * already a random distance inside the volume. It also appears to create
   * banding artifacts for unknown reasons. */
  int max_steps;
  float step_size, step_shade_offset, unused;
  volume_step_init(kg,
                   &rng_state,
                   object_step_size,
                   ray->t,
                   &step_size,
                   &step_shade_offset,
                   &unused,
                   &max_steps);
  const float steps_offset = 1.0f;

  /* compute extinction at the start */
  float t = 0.0f;

  float3 sum = zero_float3();

  for (int i = 0; i < max_steps; i++) {
    /* advance to new position */
    float new_t = min(ray->t, (i + steps_offset) * step_size);
    float dt = new_t - t;

    float3 new_P = ray->P + ray->D * (t + dt * step_shade_offset);
    float3 sigma_t = zero_float3();

    /* compute attenuation over segment */
    sd->P = new_P;
    if (shadow_volume_shader_sample(INTEGRATOR_STATE_PASS, sd, &sigma_t)) {
      /* Compute expf() only for every Nth step, to save some calculations
       * because exp(a)*exp(b) = exp(a+b), also do a quick VOLUME_THROUGHPUT_EPSILON
       * check then. */
      sum += (-sigma_t * dt);
      if ((i & 0x07) == 0) { /* ToDo: Other interval? */
        tp = *throughput * exp3(sum);

        /* stop if nearly all light is blocked */
        if (tp.x < VOLUME_THROUGHPUT_EPSILON && tp.y < VOLUME_THROUGHPUT_EPSILON &&
            tp.z < VOLUME_THROUGHPUT_EPSILON)
          break;
      }
    }

    /* stop if at the end of the volume */
    t = new_t;
    if (t == ray->t) {
      /* Update throughput in case we haven't done it above */
      tp = *throughput * exp3(sum);
      break;
    }
  }

  *throughput = tp;
}

/* Equi-angular sampling as in:
 * "Importance Sampling Techniques for Path Tracing in Participating Media" */

ccl_device float volume_equiangular_sample(const Ray *ccl_restrict ray,
                                           const float3 light_P,
                                           const float xi,
                                           float *pdf)
{
  const float t = ray->t;
  const float delta = dot((light_P - ray->P), ray->D);
  const float D = safe_sqrtf(len_squared(light_P - ray->P) - delta * delta);
  if (UNLIKELY(D == 0.0f)) {
    *pdf = 0.0f;
    return 0.0f;
  }
  const float theta_a = -atan2f(delta, D);
  const float theta_b = atan2f(t - delta, D);
  const float t_ = D * tanf((xi * theta_b) + (1 - xi) * theta_a);
  if (UNLIKELY(theta_b == theta_a)) {
    *pdf = 0.0f;
    return 0.0f;
  }
  *pdf = D / ((theta_b - theta_a) * (D * D + t_ * t_));

  return min(t, delta + t_); /* min is only for float precision errors */
}

ccl_device float volume_equiangular_pdf(const Ray *ccl_restrict ray,
                                        const float3 light_P,
                                        const float sample_t)
{
  const float delta = dot((light_P - ray->P), ray->D);
  const float D = safe_sqrtf(len_squared(light_P - ray->P) - delta * delta);
  if (UNLIKELY(D == 0.0f)) {
    return 0.0f;
  }

  const float t = ray->t;
  const float t_ = sample_t - delta;

  const float theta_a = -atan2f(delta, D);
  const float theta_b = atan2f(t - delta, D);
  if (UNLIKELY(theta_b == theta_a)) {
    return 0.0f;
  }

  const float pdf = D / ((theta_b - theta_a) * (D * D + t_ * t_));

  return pdf;
}

ccl_device float volume_equiangular_cdf(const Ray *ccl_restrict ray,
                                        const float3 light_P,
                                        const float sample_t)
{
  float delta = dot((light_P - ray->P), ray->D);
  float D = safe_sqrtf(len_squared(light_P - ray->P) - delta * delta);
  if (UNLIKELY(D == 0.0f)) {
    return 0.0f;
  }

  const float t = ray->t;
  const float t_ = sample_t - delta;

  const float theta_a = -atan2f(delta, D);
  const float theta_b = atan2f(t - delta, D);
  if (UNLIKELY(theta_b == theta_a)) {
    return 0.0f;
  }

  const float theta_sample = atan2f(t_, D);
  const float cdf = (theta_sample - theta_a) / (theta_b - theta_a);

  return cdf;
}

/* Distance sampling */

ccl_device float volume_distance_sample(
    float max_t, float3 sigma_t, int channel, float xi, float3 *transmittance, float3 *pdf)
{
  /* xi is [0, 1[ so log(0) should never happen, division by zero is
   * avoided because sample_sigma_t > 0 when SD_SCATTER is set */
  float sample_sigma_t = volume_channel_get(sigma_t, channel);
  float3 full_transmittance = volume_color_transmittance(sigma_t, max_t);
  float sample_transmittance = volume_channel_get(full_transmittance, channel);

  float sample_t = min(max_t, -logf(1.0f - xi * (1.0f - sample_transmittance)) / sample_sigma_t);

  *transmittance = volume_color_transmittance(sigma_t, sample_t);
  *pdf = safe_divide_color(sigma_t * *transmittance, one_float3() - full_transmittance);

  /* todo: optimization: when taken together with hit/miss decision,
   * the full_transmittance cancels out drops out and xi does not
   * need to be remapped */

  return sample_t;
}

ccl_device float3 volume_distance_pdf(float max_t, float3 sigma_t, float sample_t)
{
  float3 full_transmittance = volume_color_transmittance(sigma_t, max_t);
  float3 transmittance = volume_color_transmittance(sigma_t, sample_t);

  return safe_divide_color(sigma_t * transmittance, one_float3() - full_transmittance);
}

/* Emission */

ccl_device float3 volume_emission_integrate(VolumeShaderCoefficients *coeff,
                                            int closure_flag,
                                            float3 transmittance,
                                            float t)
{
  /* integral E * exp(-sigma_t * t) from 0 to t = E * (1 - exp(-sigma_t * t))/sigma_t
   * this goes to E * t as sigma_t goes to zero
   *
   * todo: we should use an epsilon to avoid precision issues near zero sigma_t */
  float3 emission = coeff->emission;

  if (closure_flag & SD_EXTINCTION) {
    float3 sigma_t = coeff->sigma_t;

    emission.x *= (sigma_t.x > 0.0f) ? (1.0f - transmittance.x) / sigma_t.x : t;
    emission.y *= (sigma_t.y > 0.0f) ? (1.0f - transmittance.y) / sigma_t.y : t;
    emission.z *= (sigma_t.z > 0.0f) ? (1.0f - transmittance.z) / sigma_t.z : t;
  }
  else
    emission *= t;

  return emission;
}

/* Volume Integration */

typedef struct VolumeIntegrateState {
  /* Volume segment extents. */
  float start_t;
  float end_t;

  /* Current throughput. */
  float3 throughput;

  /* If volume is absorption-only up to this point, and no probabilistic
   * scattering or termination has been used yet. */
  bool absorption_only;

  /* Random numbers for scattering. */
  float rscatter;
  float rphase;
} VolumeIntegrateState;

ccl_device_forceinline void volume_integrate_step_scattering(
    const ShaderData *sd,
    const VolumeShaderCoefficients &ccl_restrict coeff,
    const float3 transmittance,
    VolumeIntegrateState &ccl_restrict vstate,
    VolumeIntegrateResult &ccl_restrict result)
{
  /* Distance sampling */

  /* Pick random color channel, we use the Veach one-sample
   * model with balance heuristic for the channels. */
  const float3 albedo = safe_divide_color(coeff.sigma_s, coeff.sigma_t);
  float3 channel_pdf;
  const int channel = volume_sample_channel(
      albedo, result.indirect_throughput, vstate.rphase, &channel_pdf);

  /* decide if we will scatter or continue */
  const float sample_transmittance = volume_channel_get(transmittance, channel);

  if (1.0f - vstate.rscatter >= sample_transmittance) {
    /* compute sampling distance */
    const float sample_sigma_t = volume_channel_get(coeff.sigma_t, channel);
    const float new_dt = -logf(1.0f - vstate.rscatter) / sample_sigma_t;
    const float new_t = vstate.start_t + new_dt;

    /* transmittance and pdf */
    const float3 new_transmittance = volume_color_transmittance(coeff.sigma_t, new_dt);
    const float3 pdf = coeff.sigma_t * new_transmittance;

    /* throughput */
    result.indirect_scatter = true;
    result.indirect_t = new_t;
    result.indirect_throughput *= coeff.sigma_s * new_transmittance / dot(channel_pdf, pdf);
    shader_copy_volume_phases(&result.indirect_phases, sd);

    result.direct_scatter = true;
    result.direct_t = result.indirect_t;
    result.direct_throughput = result.indirect_throughput;
    shader_copy_volume_phases(&result.direct_phases, sd); /* TODO: only copy once? */
  }
  else {
    /* throughput */
    const float pdf = dot(channel_pdf, transmittance);
    result.indirect_throughput *= transmittance / pdf;

    /* remap rscatter so we can reuse it and keep thing stratified */
    vstate.rscatter = 1.0f - (1.0f - vstate.rscatter) / sample_transmittance;
  }
}

/* heterogeneous volume distance sampling: integrate stepping through the
 * volume until we reach the end, get absorbed entirely, or run out of
 * iterations. this does probabilistically scatter or get transmitted through
 * for path tracing where we don't want to branch. */
ccl_device_forceinline void volume_integrate_heterogeneous(INTEGRATOR_STATE_ARGS,
                                                           Ray *ccl_restrict ray,
                                                           ShaderData *ccl_restrict sd,
                                                           const RNGState *rng_state,
                                                           ccl_global float *ccl_restrict
                                                               render_buffer,
                                                           const float object_step_size,
                                                           VolumeIntegrateResult &result)
{
  /* Prepare for stepping.
   * Using a different step offset for the first step avoids banding artifacts. */
  int max_steps;
  float step_size, step_shade_offset, steps_offset;
  volume_step_init(kg,
                   rng_state,
                   object_step_size,
                   ray->t,
                   &step_size,
                   &step_shade_offset,
                   &steps_offset,
                   &max_steps);

  /* Initialize volume integration state. */
  VolumeIntegrateState vstate ccl_optional_struct_init;
  vstate.start_t = 0.0f;
  vstate.end_t = 0.0f;
  vstate.absorption_only = true;
  vstate.rscatter = path_state_rng_1D(kg, rng_state, PRNG_SCATTER_DISTANCE);
  vstate.rphase = path_state_rng_1D(kg, rng_state, PRNG_PHASE_CHANNEL);

  /* Initialize volume integration result. */
  const float3 throughput = INTEGRATOR_STATE(path, throughput);
  result.direct_throughput = throughput;
  result.indirect_throughput = throughput;

  for (int i = 0; i < max_steps; i++) {
    /* Advance to new position */
    vstate.end_t = min(ray->t, (i + steps_offset) * step_size);
    const float shade_t = vstate.start_t + (vstate.end_t - vstate.start_t) * step_shade_offset;
    sd->P = ray->P + ray->D * shade_t;

    /* compute segment */
    VolumeShaderCoefficients coeff ccl_optional_struct_init;
    if (volume_shader_sample(INTEGRATOR_STATE_PASS, sd, &coeff)) {
      const int closure_flag = sd->flag;

      /* Evaluate transmittance over segment. */
      const float dt = (vstate.end_t - vstate.start_t);
      const float3 transmittance = (closure_flag & SD_EXTINCTION) ?
                                       volume_color_transmittance(coeff.sigma_t, dt) :
                                       one_float3();

      /* Emission. */
      if (closure_flag & SD_EMISSION) {
        /* Only write emission before indirect light scatter position, since we terminate
         * stepping at that point if we have already found a direct light scatter position. */
        if (!result.indirect_scatter) {
          /* TODO: write only once to avoid overhead of atomics? */
          const float3 emission = volume_emission_integrate(
              &coeff, closure_flag, transmittance, dt);
          kernel_accum_emission(
              INTEGRATOR_STATE_PASS, result.indirect_throughput, emission, render_buffer);
        }
      }

      if (closure_flag & SD_EXTINCTION) {
        if ((closure_flag & SD_SCATTER) || !vstate.absorption_only) {
          /* Scattering and absorption. */
          volume_integrate_step_scattering(sd, coeff, transmittance, vstate, result);
        }
        else {
          /* Absorption only. */
          result.indirect_throughput *= transmittance;
          result.direct_throughput *= transmittance;
        }

        /* Stop if nearly all light blocked. */
        if (!result.indirect_scatter) {
          if (max3(result.indirect_throughput) < VOLUME_THROUGHPUT_EPSILON) {
            result.indirect_throughput = zero_float3();
            break;
          }
        }
        else if (!result.direct_scatter) {
          if (max3(result.direct_throughput) < VOLUME_THROUGHPUT_EPSILON) {
            break;
          }
        }
      }

      /* If we have scattering data for both direct and indirect, we're done. */
      if (result.direct_scatter && result.indirect_scatter) {
        break;
      }
    }

    /* Stop if at the end of the volume. */
    vstate.start_t = vstate.end_t;
    if (vstate.start_t == ray->t) {
      break;
    }
  }
}

#  ifdef __EMISSION__
/* Path tracing: sample point on light and evaluate light shader, then
 * queue shadow ray to be traced. */
ccl_device_forceinline bool integrate_volume_sample_light(INTEGRATOR_STATE_ARGS,
                                                          const ShaderData *ccl_restrict sd,
                                                          const RNGState *ccl_restrict rng_state,
                                                          LightSample *ccl_restrict ls)
{
  /* Test if there is a light or BSDF that needs direct light. */
  if (!kernel_data.integrator.use_direct_light) {
    return false;
  }

  /* Sample position on a light. */
  const int path_flag = INTEGRATOR_STATE(path, flag);
  const uint bounce = INTEGRATOR_STATE(path, bounce);
  float light_u, light_v;
  path_state_rng_2D(kg, rng_state, PRNG_LIGHT_U, &light_u, &light_v);

  light_sample(kg, light_u, light_v, sd->time, sd->P, bounce, path_flag, ls);

  if (ls->shader & SHADER_EXCLUDE_SCATTER) {
    return false;
  }

  return true;
}

/* Path tracing: sample point on light and evaluate light shader, then
 * queue shadow ray to be traced. */
ccl_device_forceinline void integrate_volume_direct_light(INTEGRATOR_STATE_ARGS,
                                                          const ShaderData *ccl_restrict sd,
                                                          const RNGState *ccl_restrict rng_state,
                                                          const float3 P,
                                                          const ShaderVolumePhases *ccl_restrict
                                                              phases,
                                                          const float3 throughput,
                                                          LightSample *ccl_restrict ls)
{
  /* Sample position on the same light again, now from the shading
   * point where we scattered.
   *
   * TODO: decorrelate random numbers and use light_sample_new_position to
   * avoid resampling the CDF. */
  {
    const int path_flag = INTEGRATOR_STATE(path, flag);
    const uint bounce = INTEGRATOR_STATE(path, bounce);
    float light_u, light_v;
    path_state_rng_2D(kg, rng_state, PRNG_LIGHT_U, &light_u, &light_v);

    if (!light_sample(kg, light_u, light_v, sd->time, P, bounce, path_flag, ls)) {
      return;
    }
  }

  /* Evaluate light shader.
   *
   * TODO: can we reuse sd memory? In theory we can move this after
   * integrate_surface_bounce, evaluate the BSDF, and only then evaluate
   * the light shader. This could also move to its own kernel, for
   * non-constant light sources. */
  ShaderDataTinyStorage emission_sd_storage;
  ShaderData *emission_sd = AS_SHADER_DATA(&emission_sd_storage);
  const float3 light_eval = light_sample_shader_eval(
      INTEGRATOR_STATE_PASS, emission_sd, ls, sd->time);
  if (is_zero(light_eval)) {
    return;
  }

  /* Evaluate BSDF. */
  BsdfEval phase_eval ccl_optional_struct_init;
  const float phase_pdf = shader_volume_phase_eval(kg, sd, phases, ls->D, &phase_eval);

  if (ls->shader & SHADER_USE_MIS) {
    float mis_weight = power_heuristic(ls->pdf, phase_pdf);
    bsdf_eval_mul(&phase_eval, mis_weight);
  }

  bsdf_eval_mul3(&phase_eval, light_eval / ls->pdf);

  /* Path termination. */
  const float terminate = path_state_rng_light_termination(kg, rng_state);
  if (light_sample_terminate(kg, ls, &phase_eval, terminate)) {
    return;
  }

  /* Create shadow ray. */
  Ray ray ccl_optional_struct_init;
  light_sample_to_volume_shadow_ray(kg, sd, ls, P, &ray);
  const bool is_light = light_sample_is_light(ls);

  /* Write shadow ray and associated state to global memory. */
  integrator_state_write_shadow_ray(INTEGRATOR_STATE_PASS, &ray);

  /* Copy state from main path to shadow path. */
  const uint16_t bounce = INTEGRATOR_STATE(path, bounce);
  const uint16_t transparent_bounce = INTEGRATOR_STATE(path, transparent_bounce);
  uint32_t shadow_flag = INTEGRATOR_STATE(path, flag);
  shadow_flag |= (is_light) ? PATH_RAY_SHADOW_FOR_LIGHT : 0;
  shadow_flag |= PATH_RAY_VOLUME_PASS;
  const float3 throughput_phase = throughput * bsdf_eval_sum(&phase_eval);

  if (kernel_data.kernel_features & KERNEL_FEATURE_LIGHT_PASSES) {
    const float3 diffuse_glossy_ratio = (bounce == 0) ?
                                            one_float3() :
                                            INTEGRATOR_STATE(path, diffuse_glossy_ratio);
    INTEGRATOR_STATE_WRITE(shadow_path, diffuse_glossy_ratio) = diffuse_glossy_ratio;
  }

  INTEGRATOR_STATE_WRITE(shadow_path, flag) = shadow_flag;
  INTEGRATOR_STATE_WRITE(shadow_path, bounce) = bounce;
  INTEGRATOR_STATE_WRITE(shadow_path, transparent_bounce) = transparent_bounce;
  INTEGRATOR_STATE_WRITE(shadow_path, throughput) = throughput_phase;

  if (kernel_data.kernel_features & KERNEL_FEATURE_SHADOW_PASS) {
    INTEGRATOR_STATE_WRITE(shadow_path, unshadowed_throughput) = throughput;
  }

  integrator_state_copy_volume_stack_to_shadow(INTEGRATOR_STATE_PASS);

  /* Branch off shadow kernel. */
  INTEGRATOR_SHADOW_PATH_INIT(DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW);
}
#  endif

/* Path tracing: scatter in new direction using phase function. */
ccl_device_forceinline bool integrate_volume_phase_scatter(INTEGRATOR_STATE_ARGS,
                                                           ShaderData *sd,
                                                           const RNGState *rng_state,
                                                           const ShaderVolumePhases *phases)
{
  float phase_u, phase_v;
  path_state_rng_2D(kg, rng_state, PRNG_BSDF_U, &phase_u, &phase_v);

  /* Phase closure, sample direction. */
  float phase_pdf;
  BsdfEval phase_eval ccl_optional_struct_init;
  float3 phase_omega_in ccl_optional_struct_init;
  differential3 phase_domega_in ccl_optional_struct_init;

  const int label = shader_volume_phase_sample(kg,
                                               sd,
                                               phases,
                                               phase_u,
                                               phase_v,
                                               &phase_eval,
                                               &phase_omega_in,
                                               &phase_domega_in,
                                               &phase_pdf);

  if (phase_pdf == 0.0f || bsdf_eval_is_zero(&phase_eval)) {
    return false;
  }

  /* Setup ray. */
  INTEGRATOR_STATE_WRITE(ray, P) = sd->P;
  INTEGRATOR_STATE_WRITE(ray, D) = normalize(phase_omega_in);
  INTEGRATOR_STATE_WRITE(ray, t) = FLT_MAX;

#  ifdef __RAY_DIFFERENTIALS__
  INTEGRATOR_STATE_WRITE(ray, dP) = differential_make_compact(sd->dP);
  INTEGRATOR_STATE_WRITE(ray, dD) = differential_make_compact(phase_domega_in);
#  endif

  /* Update throughput. */
  const float3 throughput = INTEGRATOR_STATE(path, throughput);
  const float3 throughput_phase = throughput * bsdf_eval_sum(&phase_eval) / phase_pdf;
  INTEGRATOR_STATE_WRITE(path, throughput) = throughput_phase;

  if (kernel_data.kernel_features & KERNEL_FEATURE_LIGHT_PASSES) {
    INTEGRATOR_STATE_WRITE(path, diffuse_glossy_ratio) = one_float3();
  }

  /* Update path state */
  INTEGRATOR_STATE_WRITE(path, mis_ray_pdf) = phase_pdf;
  INTEGRATOR_STATE_WRITE(path, mis_ray_t) = 0.0f;
  INTEGRATOR_STATE_WRITE(path, min_ray_pdf) = fminf(phase_pdf,
                                                    INTEGRATOR_STATE(path, min_ray_pdf));

  path_state_next(INTEGRATOR_STATE_PASS, label);
  return true;
}

/* get the volume attenuation and emission over line segment defined by
 * ray, with the assumption that there are no surfaces blocking light
 * between the endpoints. distance sampling is used to decide if we will
 * scatter or not. */
ccl_device VolumeIntegrateEvent volume_integrate(INTEGRATOR_STATE_ARGS,
                                                 Ray *ccl_restrict ray,
                                                 ccl_global float *ccl_restrict render_buffer)
{
  ShaderData sd;
  shader_setup_from_volume(kg, &sd, ray);

  /* Load random number state. */
  RNGState rng_state;
  path_state_rng_load(INTEGRATOR_STATE_PASS, &rng_state);

  /* Sample light ahead of volume stepping. */
  LightSample ls ccl_optional_struct_init;
  const bool need_light_sample = !(INTEGRATOR_STATE(path, flag) & PATH_RAY_TERMINATE);
  if (need_light_sample) {
    integrate_volume_sample_light(INTEGRATOR_STATE_PASS, &sd, &rng_state, &ls);
  }

  /* TODO: expensive to zero closures? */
  VolumeIntegrateResult result = {};

  /* Step through volume. */
  const float step_size = volume_stack_step_size(INTEGRATOR_STATE_PASS, [=](const int i) {
    return integrator_state_read_volume_stack(INTEGRATOR_STATE_PASS, i);
  });

  volume_integrate_heterogeneous(
      INTEGRATOR_STATE_PASS, ray, &sd, &rng_state, render_buffer, step_size, result);

  /* Perform path termination. The intersect_closest will have already marked this path
   * to be terminated. That will shading evaluating to leave out any scattering closures,
   * but emission and absorption are still handled for multiple importance sampling. */
  const uint32_t path_flag = INTEGRATOR_STATE(path, flag);
  const float probability = (path_flag & PATH_RAY_TERMINATE_IN_NEXT_VOLUME) ?
                                0.0f :
                                path_state_continuation_probability(INTEGRATOR_STATE_PASS,
                                                                    path_flag);
  if (probability == 0.0f) {
    return VOLUME_PATH_MISSED;
  }

  /* Direct light. */
  if (result.direct_scatter) {
    const float3 direct_P = ray->P + result.direct_t * ray->D;
    result.direct_throughput /= probability;
    integrate_volume_direct_light(INTEGRATOR_STATE_PASS,
                                  &sd,
                                  &rng_state,
                                  direct_P,
                                  &result.direct_phases,
                                  result.direct_throughput,
                                  &ls);
  }

  /* Indirect light.
   *
   * Only divide throughput by probability if we scatter. For the attenuation
   * case the next surface will already do this division. */
  if (result.indirect_scatter) {
    result.indirect_throughput /= probability;
  }
  INTEGRATOR_STATE_WRITE(path, throughput) = result.indirect_throughput;

  if (result.indirect_scatter) {
    sd.P = ray->P + result.indirect_t * ray->D;

    if (integrate_volume_phase_scatter(
            INTEGRATOR_STATE_PASS, &sd, &rng_state, &result.indirect_phases)) {
      return VOLUME_PATH_SCATTERED;
    }
    else {
      return VOLUME_PATH_MISSED;
    }
  }
  else {
    return VOLUME_PATH_ATTENUATED;
  }
}

#endif

ccl_device void integrator_shade_volume(INTEGRATOR_STATE_ARGS,
                                        ccl_global float *ccl_restrict render_buffer)
{
#ifdef __VOLUME__
  /* Setup shader data. */
  Ray ray ccl_optional_struct_init;
  integrator_state_read_ray(INTEGRATOR_STATE_PASS, &ray);

  Intersection isect ccl_optional_struct_init;
  integrator_state_read_isect(INTEGRATOR_STATE_PASS, &isect);

  /* Set ray length to current segment. */
  ray.t = (isect.prim != PRIM_NONE) ? isect.t : FLT_MAX;

  /* Clean volume stack for background rays. */
  if (isect.prim == PRIM_NONE) {
    volume_stack_clean(INTEGRATOR_STATE_PASS);
  }

  VolumeIntegrateEvent event = volume_integrate(INTEGRATOR_STATE_PASS, &ray, render_buffer);

  if (event == VOLUME_PATH_SCATTERED) {
    /* Queue intersect_closest kernel. */
    INTEGRATOR_PATH_NEXT(DEVICE_KERNEL_INTEGRATOR_SHADE_VOLUME,
                         DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST);
    return;
  }
  else if (event == VOLUME_PATH_MISSED) {
    /* End path. */
    INTEGRATOR_PATH_TERMINATE(DEVICE_KERNEL_INTEGRATOR_SHADE_VOLUME);
    return;
  }
  else {
    /* Continue to background, light or surface. */
    if (isect.prim == PRIM_NONE) {
      INTEGRATOR_PATH_NEXT(DEVICE_KERNEL_INTEGRATOR_SHADE_VOLUME,
                           DEVICE_KERNEL_INTEGRATOR_SHADE_BACKGROUND);
      return;
    }
    else if (isect.type & PRIMITIVE_LAMP) {
      INTEGRATOR_PATH_NEXT(DEVICE_KERNEL_INTEGRATOR_SHADE_VOLUME,
                           DEVICE_KERNEL_INTEGRATOR_SHADE_LIGHT);
      return;
    }
    else {
      /* Hit a surface, continue with surface kernel unless terminated. */
      const int shader = intersection_get_shader(kg, &isect);
      const int flags = kernel_tex_fetch(__shaders, shader).flags;

      integrator_intersect_shader_next_kernel<DEVICE_KERNEL_INTEGRATOR_SHADE_VOLUME>(
          INTEGRATOR_STATE_PASS, &isect, shader, flags);
      return;
    }
  }
#endif /* __VOLUME__ */
}

CCL_NAMESPACE_END
