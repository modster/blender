
#pragma BLENDER_REQUIRE(common_utiltex_lib.glsl)
#pragma BLENDER_REQUIRE(lights_lib.glsl)
#pragma BLENDER_REQUIRE(lightprobe_lib.glsl)
#pragma BLENDER_REQUIRE(ambient_occlusion_lib.glsl)
#pragma BLENDER_REQUIRE(ssr_lib.glsl)

/**
 * Extensive use of Macros to be able to change the maximum amount of evaluated closure easily.
 * NOTE: GLSL does not support variadic macros.
 *
 * Example
 * // Declare the cl_eval function
 * CLOSURE_EVAL_FUNCTION_DECLARE_3(name, Diffuse, Glossy, Refraction);
 * // Declare the inputs & outputs
 * CLOSURE_VARS_DECLARE(Diffuse, Glossy, Refraction);
 * // Specify inputs
 * in_Diffuse_0.N = N;
 * ...
 * // Call the cl_eval function
 * CLOSURE_EVAL_FUNCTION_3(name, Diffuse, Glossy, Refraction);
 * // Get the cl_out
 * closure.radiance = out_Diffuse_0.radiance + out_Glossy_1.radiance + out_Refraction_2.radiance;
 **/

#define CLOSURE_VARS_DECLARE(t0, t1, t2, t3) \
  ClosureInput##t0 in_##t0##_0 = CLOSURE_INPUT_##t0##_DEFAULT; \
  ClosureInput##t1 in_##t1##_1 = CLOSURE_INPUT_##t1##_DEFAULT; \
  ClosureInput##t2 in_##t2##_2 = CLOSURE_INPUT_##t2##_DEFAULT; \
  ClosureInput##t3 in_##t3##_3 = CLOSURE_INPUT_##t3##_DEFAULT; \
  ClosureOutput##t0 out_##t0##_0; \
  ClosureOutput##t1 out_##t1##_1; \
  ClosureOutput##t2 out_##t2##_2; \
  ClosureOutput##t3 out_##t3##_3;

#define CLOSURE_EVAL_DECLARE(t0, t1, t2, t3) \
  ClosureEval##t0 eval_##t0##_0 = closure_##t0##_eval_init(in_##t0##_0, cl_common, out_##t0##_0); \
  ClosureEval##t1 eval_##t1##_1 = closure_##t1##_eval_init(in_##t1##_1, cl_common, out_##t1##_1); \
  ClosureEval##t2 eval_##t2##_2 = closure_##t2##_eval_init(in_##t2##_2, cl_common, out_##t2##_2); \
  ClosureEval##t3 eval_##t3##_3 = closure_##t3##_eval_init(in_##t3##_3, cl_common, out_##t3##_3);

#define CLOSURE_META_SUBROUTINE(subroutine, t0, t1, t2, t3) \
  closure_##t0##_##subroutine(in_##t0##_0, eval_##t0##_0, cl_common, out_##t0##_0); \
  closure_##t1##_##subroutine(in_##t1##_1, eval_##t1##_1, cl_common, out_##t1##_1); \
  closure_##t2##_##subroutine(in_##t2##_2, eval_##t2##_2, cl_common, out_##t2##_2); \
  closure_##t3##_##subroutine(in_##t3##_3, eval_##t3##_3, cl_common, out_##t3##_3);

#define CLOSURE_META_SUBROUTINE_DATA(subroutine, sub_data, t0, t1, t2, t3) \
  closure_##t0##_##subroutine(in_##t0##_0, eval_##t0##_0, cl_common, sub_data, out_##t0##_0); \
  closure_##t1##_##subroutine(in_##t1##_1, eval_##t1##_1, cl_common, sub_data, out_##t1##_1); \
  closure_##t2##_##subroutine(in_##t2##_2, eval_##t2##_2, cl_common, sub_data, out_##t2##_2); \
  closure_##t3##_##subroutine(in_##t3##_3, eval_##t3##_3, cl_common, sub_data, out_##t3##_3);

/* Inputs are inout so that callers can get the final inputs used for evaluation. */
#define CLOSURE_EVAL_FUNCTION_DECLARE(name, t0, t1, t2, t3) \
  void closure_##name##_eval(inout ClosureInput##t0 in_##t0##_0, \
                             inout ClosureInput##t1 in_##t1##_1, \
                             inout ClosureInput##t2 in_##t2##_2, \
                             inout ClosureInput##t3 in_##t3##_3, \
                             out ClosureOutput##t0 out_##t0##_0, \
                             out ClosureOutput##t1 out_##t1##_1, \
                             out ClosureOutput##t2 out_##t2##_2, \
                             out ClosureOutput##t3 out_##t3##_3) \
  { \
    ClosureEvalCommon cl_common = closure_Common_eval_init(); \
    CLOSURE_EVAL_DECLARE(t0, t1, t2, t3); \
\
    ClosurePlanarData planar; \
    PLANAR_ITER_BEGIN(planar) \
    { \
      CLOSURE_META_SUBROUTINE_DATA(planar_eval, planar, t0, t1, t2, t3); \
    } \
    PLANAR_ITER_END \
\
    ClosureCubemapData cube; \
    CUBEMAP_ITER_BEGIN(cube) \
    { \
      CLOSURE_META_SUBROUTINE_DATA(cubemap_eval, cube, t0, t1, t2, t3); \
    } \
    CUBEMAP_ITER_END \
\
    ClosureGridData grid; \
    GRID_ITER_BEGIN(grid) \
    { \
      CLOSURE_META_SUBROUTINE_DATA(grid_eval, grid, t0, t1, t2, t3); \
    } \
    GRID_ITER_END \
\
    CLOSURE_META_SUBROUTINE(indirect_end, t0, t1, t2, t3); \
\
    ClosureLightData light; \
    LIGHT_ITER_BEGIN(light) \
    { \
      CLOSURE_META_SUBROUTINE_DATA(light_eval, light, t0, t1, t2, t3); \
    } \
    LIGHT_ITER_END \
\
    CLOSURE_META_SUBROUTINE(eval_end, t0, t1, t2, t3); \
  }

#define CLOSURE_EVAL_FUNCTION(name, t0, t1, t2, t3) \
  closure_##name##_eval(in_##t0##_0, \
                        in_##t1##_1, \
                        in_##t2##_2, \
                        in_##t3##_3, \
                        out_##t0##_0, \
                        out_##t1##_1, \
                        out_##t2##_2, \
                        out_##t3##_3)

#define CLOSURE_EVAL_FUNCTION_DECLARE_1(name, t0) \
  CLOSURE_EVAL_FUNCTION_DECLARE(name, t0, Dummy, Dummy, Dummy)
#define CLOSURE_EVAL_FUNCTION_DECLARE_2(name, t0, t1) \
  CLOSURE_EVAL_FUNCTION_DECLARE(name, t0, t1, Dummy, Dummy)
#define CLOSURE_EVAL_FUNCTION_DECLARE_3(name, t0, t1, t2) \
  CLOSURE_EVAL_FUNCTION_DECLARE(name, t0, t1, t2, Dummy)
#define CLOSURE_EVAL_FUNCTION_DECLARE_4(name, t0, t1, t2, t3) \
  CLOSURE_EVAL_FUNCTION_DECLARE(name, t0, t1, t2, t3)

#define CLOSURE_VARS_DECLARE_1(t0) CLOSURE_VARS_DECLARE(t0, Dummy, Dummy, Dummy)
#define CLOSURE_VARS_DECLARE_2(t0, t1) CLOSURE_VARS_DECLARE(t0, t1, Dummy, Dummy)
#define CLOSURE_VARS_DECLARE_3(t0, t1, t2) CLOSURE_VARS_DECLARE(t0, t1, t2, Dummy)
#define CLOSURE_VARS_DECLARE_4(t0, t1, t2, t3) CLOSURE_VARS_DECLARE(t0, t1, t2, t3)

#define CLOSURE_EVAL_FUNCTION_1(name, t0) CLOSURE_EVAL_FUNCTION(name, t0, Dummy, Dummy, Dummy)
#define CLOSURE_EVAL_FUNCTION_2(name, t0, t1) CLOSURE_EVAL_FUNCTION(name, t0, t1, Dummy, Dummy)
#define CLOSURE_EVAL_FUNCTION_3(name, t0, t1, t2) CLOSURE_EVAL_FUNCTION(name, t0, t1, t2, Dummy)
#define CLOSURE_EVAL_FUNCTION_4(name, t0, t1, t2, t3) CLOSURE_EVAL_FUNCTION(name, t0, t1, t2, t3)

/* -------------------------------------------------------------------- */
/** \name Common cl_eval data
 *
 * Eval data not dependant on input parameters. All might not be used but unused ones
 * will be optimized out.
 * \{ */

struct ClosureEvalCommon {
  vec3 V;    /** View vector. */
  vec3 P;    /** Surface position. */
  vec3 N;    /** Normal vector, always facing camera. */
  vec3 vN;   /** Normal vector, always facing camera. (viewspace) */
  vec3 vP;   /** Surface position. (viewspace) */
  vec3 vNg;  /** Geometric normal, always facing camera. (viewspace) */
  vec4 rand; /** Random numbers. 3 random sequences. zw is a random point on a circle. */

  float specular_accum; /** Specular probe accumulator. Shared between planar and cubemap probe. */
  float diffuse_accum;  /** Diffuse probe accumulator. */
  float tracing_depth;  /** Viewspace depth to start raytracing from. */
};

/* Common cl_out struct used by most closures. */
struct ClosureOutput {
  vec3 radiance;
};

ClosureEvalCommon closure_Common_eval_init(void)
{
  ClosureEvalCommon cl_eval;
  cl_eval.rand = texelfetch_noise_tex(gl_FragCoord.xy);
  cl_eval.V = cameraVec;
  cl_eval.P = worldPosition;
  cl_eval.N = safe_normalize(gl_FrontFacing ? worldNormal : -worldNormal);
  cl_eval.vN = safe_normalize(gl_FrontFacing ? viewNormal : -viewNormal);
  cl_eval.vP = viewPosition;
  cl_eval.vNg = safe_normalize(cross(dFdx(viewPosition), dFdy(viewPosition)));
  /* TODO(fclem) See if we can avoid this complicated setup. */
  cl_eval.tracing_depth = gl_FragCoord.z;
  /* Constant bias (due to depth buffer precision) */
  /* Magic numbers for 24bits of precision.
   * From http://terathon.com/gdc07_lengyel.pdf (slide 26) */
  cl_eval.tracing_depth -= mix(2.4e-7, 4.8e-7, gl_FragCoord.z);
  /* Convert to view Z. */
  cl_eval.tracing_depth = get_view_z_from_depth(cl_eval.tracing_depth);

  cl_eval.specular_accum = 1.0;
  cl_eval.diffuse_accum = 1.0;
  return cl_eval;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Light Loop
 *
 * \{ */

struct ClosureLightData {
  LightData data; /** Light Data. */
  vec4 L;         /** Non-Normalized Light Vector (surface to light) with length in W component. */
  float vis;      /** Light visibility. */
  float contact_shadow; /** Result of contact shadow tracing. */
};

ClosureLightData closure_light_eval_init(ClosureEvalCommon cl_common, int light_id)
{
  ClosureLightData light;
  light.data = lights_data[light_id];

  light.L.xyz = light.data.l_position - cl_common.P;
  light.L.w = length(light.L.xyz);

  light.vis = light_visibility(light.data, cl_common.P, light.L);
  light.contact_shadow = light_contact_shadows(light.data,
                                               cl_common.P,
                                               cl_common.vP,
                                               cl_common.tracing_depth,
                                               cl_common.vNg,
                                               cl_common.rand.x,
                                               light.vis);

  return light;
}

#define LIGHT_ITER_BEGIN(light) \
  for (int i = 0; i < laNumLight && i < MAX_LIGHT; i++) { \
    light = closure_light_eval_init(cl_common, i); \
    if (light.vis < 1e-8) { \
      continue; \
    }

#define LIGHT_ITER_END }

/** \} */

/* -------------------------------------------------------------------- */
/** \name Glossy Probe Loop
 *
 * \{ */

struct ClosureCubemapData {
  int id;            /** Probe id. */
  float attenuation; /** Attenuation. */
};

ClosureCubemapData closure_cubemap_eval_init(int cube_id, inout ClosureEvalCommon cl_common)
{
  ClosureCubemapData cube;
  cube.id = cube_id;
  cube.attenuation = probe_attenuation_cube(cube_id, cl_common.P);
  cube.attenuation = min(cube.attenuation, cl_common.specular_accum);
  cl_common.specular_accum -= cube.attenuation;
  return cube;
}

#define CUBEMAP_ITER_BEGIN(cube) \
  /* Starts at 1 because 0 is world cubemap. */ \
  for (int i = 1; cl_common.specular_accum > 0.0 && i < prbNumRenderCube && i < MAX_PROBE; i++) { \
    cube = closure_cubemap_eval_init(i, cl_common); \
    if (cube.attenuation < 1e-8) { \
      continue; \
    }

#define CUBEMAP_ITER_END }

/** \} */

/* -------------------------------------------------------------------- */
/** \name Glossy Planar probe Loop
 *
 * Should be run first, as it is replace by the SSR pass if SSR is enabled.
 * \{ */

struct ClosurePlanarData {
  int id;            /** Probe id. */
  PlanarData data;   /** planars_data[id]. */
  float attenuation; /** Attenuation. */
};

ClosurePlanarData closure_planar_eval_init(int planar_id, inout ClosureEvalCommon cl_common)
{
  ClosurePlanarData planar;
  planar.id = planar_id;
  planar.data = planars_data[planar_id];
  planar.attenuation = probe_attenuation_planar(planar.data, cl_common.P, cl_common.N, 0.0);
  planar.attenuation = min(planar.attenuation, cl_common.specular_accum);
  cl_common.specular_accum -= planar.attenuation;
  return planar;
}

#define PLANAR_ITER_BEGIN(planar) \
  /* Starts at 1 because 0 is world probe */ \
  for (int i = 1; cl_common.specular_accum > 0.0 && i < prbNumPlanar && i < MAX_PLANAR; i++) { \
    planar = closure_planar_eval_init(i, cl_common); \
    if (planar.attenuation < 1e-8) { \
      continue; \
    }

#define PLANAR_ITER_END }

/** \} */

/* -------------------------------------------------------------------- */
/** \name Irradiance Grid Loop
 *
 * \{ */

struct ClosureGridData {
  int id;            /** Grid id. */
  GridData data;     /** grids_data[id] */
  float attenuation; /** Attenuation. */
  vec3 local_pos;    /** Local position inside the grid. */
};

ClosureGridData closure_grid_eval_init(int id, inout ClosureEvalCommon cl_common)
{
  ClosureGridData grid;
  grid.id = id;
  grid.data = grids_data[id];
  grid.attenuation = probe_attenuation_grid(grid.data, cl_common.P, grid.local_pos);
  grid.attenuation = min(grid.attenuation, cl_common.diffuse_accum);
  cl_common.diffuse_accum -= grid.attenuation;
  return grid;
}

#define GRID_ITER_BEGIN(grid) \
  /* Starts at 1 because 0 is world irradiance. */ \
  for (int i = 1; cl_common.diffuse_accum > 0.0 && i < prbNumRenderGrid && i < MAX_GRID; i++) { \
    grid = closure_grid_eval_init(i, cl_common); \
    if (grid.attenuation < 1e-8) { \
      continue; \
    }

#define GRID_ITER_END }

/** \} */

/* -------------------------------------------------------------------- */
/** \name Dummy Closure
 *
 * Dummy closure type that will be optimized out by the compiler.
 * \{ */

#define ClosureInputDummy ClosureOutput
#define ClosureOutputDummy ClosureOutput
#define ClosureEvalDummy ClosureOutput
#define CLOSURE_EVAL_DUMMY ClosureOutput(vec3(0))
#define CLOSURE_INPUT_Dummy_DEFAULT CLOSURE_EVAL_DUMMY
#define closure_Dummy_eval_init(cl_in, cl_common, cl_out) CLOSURE_EVAL_DUMMY
#define closure_Dummy_planar_eval(cl_in, cl_eval, cl_common, data, cl_out)
#define closure_Dummy_cubemap_eval(cl_in, cl_eval, cl_common, data, cl_out)
#define closure_Dummy_grid_eval(cl_in, cl_eval, cl_common, data, cl_out)
#define closure_Dummy_indirect_end(cl_in, cl_eval, cl_common, cl_out)
#define closure_Dummy_light_eval(cl_in, cl_eval, cl_common, data, cl_out)
#define closure_Dummy_eval_end(cl_in, cl_eval, cl_common, cl_out)

/** \} */

/* -------------------------------------------------------------------- */
/** \name Glossy Closure
 * \{ */

struct ClosureInputGlossy {
  vec3 N;          /** Shading normal. */
  float roughness; /** Input roughness, not squared. */
};

#define CLOSURE_INPUT_Glossy_DEFAULT ClosureInputGlossy(vec3(0.0), 0.0)

struct ClosureEvalGlossy {
  vec4 ltc_mat;            /** LTC matrix values. */
  float ltc_brdf_scale;    /** LTC BRDF scaling. */
  vec3 probe_sampling_dir; /** Direction to sample probes from. */
};

/* Stubs. */
#define ClosureOutputGlossy ClosureOutput
#define closure_Glossy_grid_eval(cl_in, cl_eval, cl_common, data, cl_out)

#ifdef STEP_RESOLVE /* SSR */
/* Prototype. */
void ssr_resolve(ClosureInputGlossy cl_in,
                 inout ClosureEvalCommon cl_common,
                 inout ClosureOutputGlossy cl_out);
#endif

ClosureEvalGlossy closure_Glossy_eval_init(inout ClosureInputGlossy cl_in,
                                           inout ClosureEvalCommon cl_common,
                                           out ClosureOutputGlossy cl_out)
{
  cl_in.N = safe_normalize(cl_in.N);
  cl_in.roughness = clamp(cl_in.roughness, 1e-8, 0.9999);
  cl_out.radiance = vec3(0.0);

#ifdef STEP_RESOLVE /* SSR */
  ssr_resolve(cl_in, cl_common, cl_out);
#endif

  float NV = dot(cl_in.N, cl_common.V);
  vec2 lut_uv = lut_coords_ltc(NV, cl_in.roughness);

  ClosureEvalGlossy cl_eval;
  cl_eval.ltc_mat = texture(utilTex, vec3(lut_uv, LTC_MAT_LAYER));
  cl_eval.probe_sampling_dir = specular_dominant_dir(cl_in.N, cl_common.V, sqr(cl_in.roughness));

  /* The brdf split sum LUT is applied after the radiance accumulation.
   * Correct the LTC so that its energy is constant. */
  /* TODO(fclem) Optimize this so that only one scale factor is stored. */
  vec4 ltc_brdf = texture(utilTex, vec3(lut_uv, LTC_BRDF_LAYER)).barg;
  vec2 split_sum_brdf = ltc_brdf.zw;
  cl_eval.ltc_brdf_scale = (ltc_brdf.x + ltc_brdf.y) / (split_sum_brdf.x + split_sum_brdf.y);
  return cl_eval;
}

void closure_Glossy_light_eval(ClosureInputGlossy cl_in,
                               ClosureEvalGlossy cl_eval,
                               ClosureEvalCommon cl_common,
                               ClosureLightData light,
                               inout ClosureOutputGlossy cl_out)
{
  float radiance = light_specular(light.data, cl_eval.ltc_mat, cl_in.N, cl_common.V, light.L);
  radiance *= cl_eval.ltc_brdf_scale;
  cl_out.radiance += light.data.l_color *
                     (light.data.l_spec * light.vis * light.contact_shadow * radiance);
}

void closure_Glossy_planar_eval(ClosureInputGlossy cl_in,
                                ClosureEvalGlossy cl_eval,
                                ClosureEvalCommon cl_common,
                                ClosurePlanarData planar,
                                inout ClosureOutputGlossy cl_out)
{
#ifndef STEP_RESOLVE /* SSR already evaluates planar reflections. */
  vec3 probe_radiance = probe_evaluate_planar(
      planar.id, planar.data, cl_common.P, cl_in.N, cl_common.V, cl_in.roughness);
  cl_out.radiance += planar.attenuation * probe_radiance;
#endif
}

void closure_Glossy_cubemap_eval(ClosureInputGlossy cl_in,
                                 ClosureEvalGlossy cl_eval,
                                 ClosureEvalCommon cl_common,
                                 ClosureCubemapData cube,
                                 inout ClosureOutputGlossy cl_out)
{
  vec3 probe_radiance = probe_evaluate_cube(
      cube.id, cl_common.P, cl_eval.probe_sampling_dir, cl_in.roughness);
  cl_out.radiance += cube.attenuation * probe_radiance;
}

void closure_Glossy_indirect_end(ClosureInputGlossy cl_in,
                                 ClosureEvalGlossy cl_eval,
                                 ClosureEvalCommon cl_common,
                                 inout ClosureOutputGlossy cl_out)
{
  /* If not enough light has been accumulated from probes, use the world specular cubemap
   * to fill the remaining energy needed. */
  if (specToggle && cl_common.specular_accum > 0.0) {
    vec3 probe_radiance = probe_evaluate_world_spec(cl_eval.probe_sampling_dir, cl_in.roughness);
    cl_out.radiance += cl_common.specular_accum * probe_radiance;
  }

  /* TODO(fclem) Apply occlusion. */
}

void closure_Glossy_eval_end(ClosureInputGlossy cl_in,
                             ClosureEvalGlossy cl_eval,
                             ClosureEvalCommon cl_common,
                             inout ClosureOutputGlossy cl_out)
{
#if defined(DEPTH_SHADER) || defined(WORLD_BACKGROUND)
  /* This makes shader resources become unused and avoid issues with samplers. (see T59747) */
  cl_out.radiance = vec3(0.0);
  return;
#endif

  if (!specToggle) {
    cl_out.radiance = vec3(0.0);
  }
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Refraction Closure
 * \{ */

struct ClosureInputRefraction {
  vec3 N;          /** Shading normal. */
  float roughness; /** Input roughness, not squared. */
  float ior;       /** Index of refraction ratio. */
};

#define CLOSURE_INPUT_Refraction_DEFAULT ClosureInputRefraction(vec3(0.0), 0.0, 0.0)

struct ClosureEvalRefraction {
  vec3 P;                  /** LTC matrix values. */
  vec3 ltc_brdf;           /** LTC BRDF values. */
  vec3 probe_sampling_dir; /** Direction to sample probes from. */
  float probes_weight;     /** Factor to apply to probe radiance. */
};

/* Stubs. */
#define ClosureOutputRefraction ClosureOutput
#define closure_Refraction_grid_eval(cl_in, cl_eval, cl_common, data, cl_out)

ClosureEvalRefraction closure_Refraction_eval_init(inout ClosureInputRefraction cl_in,
                                                   ClosureEvalCommon cl_common,
                                                   out ClosureOutputRefraction cl_out)
{
  cl_in.N = safe_normalize(cl_in.N);
  cl_in.roughness = clamp(cl_in.roughness, 1e-8, 0.9999);
  cl_in.ior = max(cl_in.ior, 1e-5);
  cl_out.radiance = vec3(0.0);

  ClosureEvalRefraction cl_eval;
  vec3 cl_V;
  float eval_ior;
  /* Refract the view vector using the depth heuristic.
   * Then later Refract a second time the already refracted
   * ray using the inverse ior. */
  if (refractionDepth > 0.0) {
    eval_ior = 1.0 / cl_in.ior;
    cl_V = -refract(-cl_common.V, cl_in.N, eval_ior);
    vec3 plane_pos = cl_common.P - cl_in.N * refractionDepth;
    cl_eval.P = line_plane_intersect(cl_common.P, cl_V, plane_pos, cl_in.N);
  }
  else {
    eval_ior = cl_in.ior;
    cl_V = cl_common.V;
    cl_eval.P = cl_common.P;
  }

  cl_eval.probe_sampling_dir = refraction_dominant_dir(cl_in.N, cl_V, cl_in.roughness, eval_ior);
  cl_eval.probes_weight = 1.0;

#ifdef USE_REFRACTION
  if (ssrefractToggle && cl_in.roughness < ssrMaxRoughness + 0.2) {
    /* Find approximated position of the 2nd refraction event. */
    vec3 vP = (refractionDepth > 0.0) ? transform_point(ViewMatrix, cl_eval.P) : cl_common.vP;
    vec4 ssr_output = screen_space_refraction(
        vP, cl_in.N, cl_V, eval_ior, sqr(cl_in.roughness), cl_common.rand);
    ssr_output.a *= smoothstep(ssrMaxRoughness + 0.2, ssrMaxRoughness, cl_in.roughness);
    cl_out.radiance += ssr_output.rgb * ssr_output.a;
    cl_eval.probes_weight -= ssr_output.a;
  }
#endif
  return cl_eval;
}

void closure_Refraction_light_eval(ClosureInputRefraction cl_in,
                                   ClosureEvalRefraction cl_eval,
                                   ClosureEvalCommon cl_common,
                                   ClosureLightData light,
                                   inout ClosureOutputRefraction cl_out)
{
  /* Not implemented yet. */
}

void closure_Refraction_planar_eval(ClosureInputRefraction cl_in,
                                    ClosureEvalRefraction cl_eval,
                                    ClosureEvalCommon cl_common,
                                    ClosurePlanarData planar,
                                    inout ClosureOutputRefraction cl_out)
{
  /* Not implemented yet. */
}

void closure_Refraction_cubemap_eval(ClosureInputRefraction cl_in,
                                     ClosureEvalRefraction cl_eval,
                                     ClosureEvalCommon cl_common,
                                     ClosureCubemapData cube,
                                     inout ClosureOutputRefraction cl_out)
{
  vec3 probe_radiance = probe_evaluate_cube(
      cube.id, cl_eval.P, cl_eval.probe_sampling_dir, sqr(cl_in.roughness));
  cl_out.radiance += (cube.attenuation * cl_eval.probes_weight) * probe_radiance;
}

void closure_Refraction_indirect_end(ClosureInputRefraction cl_in,
                                     ClosureEvalRefraction cl_eval,
                                     ClosureEvalCommon cl_common,
                                     inout ClosureOutputRefraction cl_out)
{
  /* If not enough light has been accumulated from probes, use the world specular cubemap
   * to fill the remaining energy needed. */
  if (specToggle && cl_common.specular_accum > 0.0) {
    vec3 probe_radiance = probe_evaluate_world_spec(cl_eval.probe_sampling_dir,
                                                    sqr(cl_in.roughness));
    cl_out.radiance += (cl_common.specular_accum * cl_eval.probes_weight) * probe_radiance;
  }
}

void closure_Refraction_eval_end(ClosureInputRefraction cl_in,
                                 ClosureEvalRefraction cl_eval,
                                 ClosureEvalCommon cl_common,
                                 inout ClosureOutputRefraction cl_out)
{
#if defined(DEPTH_SHADER) || defined(WORLD_BACKGROUND)
  /* This makes shader resources become unused and avoid issues with samplers. (see T59747) */
  cl_out.radiance = vec3(0.0);
  return;
#endif

  if (!specToggle) {
    cl_out.radiance = vec3(0.0);
  }
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Diffuse Closure
 * \{ */

struct ClosureInputDiffuse {
  vec3 N;      /** Shading normal. */
  vec3 albedo; /** Used for multibounce GTAO approximation. Not applied to final radiance. */
};

#define CLOSURE_INPUT_Diffuse_DEFAULT ClosureInputDiffuse(vec3(0.0), vec3(0.0))

struct ClosureEvalDiffuse {
  vec3 bent_normal;        /** Normal pointing in the least occluded direction. */
  float ambient_occlusion; /** Final occlusion factor. */
};

/* Stubs. */
#define ClosureOutputDiffuse ClosureOutput
#define closure_Diffuse_planar_eval(cl_in, cl_eval, cl_common, data, cl_out)
#define closure_Diffuse_cubemap_eval(cl_in, cl_eval, cl_common, data, cl_out)

ClosureEvalDiffuse closure_Diffuse_eval_init(inout ClosureInputDiffuse cl_in,
                                             ClosureEvalCommon cl_common,
                                             out ClosureOutputDiffuse cl_out)
{
  cl_in.N = safe_normalize(cl_in.N);
  cl_out.radiance = vec3(0.0);

  ClosureEvalDiffuse cl_eval;
  float user_ao = 1.0; /* TODO(fclem) wire the real one through ClosureEvalCommon. */
  cl_eval.ambient_occlusion = occlusion_compute(
      cl_in.N, cl_common.vP, user_ao, cl_common.rand, cl_eval.bent_normal);
  return cl_eval;
}

void closure_Diffuse_light_eval(ClosureInputDiffuse cl_in,
                                ClosureEvalDiffuse cl_eval,
                                ClosureEvalCommon cl_common,
                                ClosureLightData light,
                                inout ClosureOutputDiffuse cl_out)
{
  float radiance = light_diffuse(light.data, cl_in.N, cl_common.V, light.L);
  /* TODO(fclem) We could try to shadow lights that are shadowless with the ambient_occlusion
   * factor here. */
  cl_out.radiance += light.data.l_color * (light.vis * light.contact_shadow * radiance);
}

void closure_Diffuse_grid_eval(ClosureInputDiffuse cl_in,
                               ClosureEvalDiffuse cl_eval,
                               ClosureEvalCommon cl_common,
                               ClosureGridData grid,
                               inout ClosureOutputDiffuse cl_out)
{
  vec3 probe_radiance = probe_evaluate_grid(
      grid.data, cl_common.P, cl_eval.bent_normal, grid.local_pos);
  cl_out.radiance += grid.attenuation * probe_radiance;
}

void closure_Diffuse_indirect_end(ClosureInputDiffuse cl_in,
                                  ClosureEvalDiffuse cl_eval,
                                  ClosureEvalCommon cl_common,
                                  inout ClosureOutputDiffuse cl_out)
{
  /* If not enough light has been accumulated from probes, use the world specular cubemap
   * to fill the remaining energy needed. */
  if (cl_common.diffuse_accum > 0.0) {
    vec3 probe_radiance = probe_evaluate_world_diff(cl_eval.bent_normal);
    cl_out.radiance += cl_common.diffuse_accum * probe_radiance;
  }
  /* Apply occlusion on radiance before the light loop. */
  cl_out.radiance *= gtao_multibounce(cl_eval.ambient_occlusion, cl_in.albedo);
}

void closure_Diffuse_eval_end(ClosureInputDiffuse cl_in,
                              ClosureEvalDiffuse cl_eval,
                              ClosureEvalCommon cl_common,
                              inout ClosureOutputDiffuse cl_out)
{
#if defined(DEPTH_SHADER) || defined(WORLD_BACKGROUND)
  /* This makes shader resources become unused and avoid issues with samplers. (see T59747) */
  cl_out.radiance = vec3(0.0);
  return;
#endif
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Translucent Closure
 * \{ */

struct ClosureInputTranslucent {
  vec3 N; /** Shading normal. */
};

#define CLOSURE_INPUT_Translucent_DEFAULT ClosureInputTranslucent(vec3(0.0))

/* Stubs. */
#define ClosureEvalTranslucent ClosureEvalDummy
#define ClosureOutputTranslucent ClosureOutput
#define closure_Translucent_planar_eval(cl_in, cl_eval, cl_common, data, cl_out)
#define closure_Translucent_cubemap_eval(cl_in, cl_eval, cl_common, data, cl_out)

ClosureEvalTranslucent closure_Translucent_eval_init(inout ClosureInputTranslucent cl_in,
                                                     ClosureEvalCommon cl_common,
                                                     out ClosureOutputTranslucent cl_out)
{
  cl_in.N = safe_normalize(cl_in.N);
  cl_out.radiance = vec3(0.0);
  return CLOSURE_EVAL_DUMMY;
}

void closure_Translucent_light_eval(ClosureInputTranslucent cl_in,
                                    ClosureEvalTranslucent cl_eval,
                                    ClosureEvalCommon cl_common,
                                    ClosureLightData light,
                                    inout ClosureOutputTranslucent cl_out)
{
  float radiance = light_diffuse(light.data, cl_in.N, cl_common.V, light.L);
  cl_out.radiance += light.data.l_color * (light.vis * radiance);
}

void closure_Translucent_grid_eval(ClosureInputTranslucent cl_in,
                                   ClosureEvalTranslucent cl_eval,
                                   ClosureEvalCommon cl_common,
                                   ClosureGridData grid,
                                   inout ClosureOutputTranslucent cl_out)
{
  vec3 probe_radiance = probe_evaluate_grid(grid.data, cl_common.P, cl_in.N, grid.local_pos);
  cl_out.radiance += grid.attenuation * probe_radiance;
}

void closure_Translucent_indirect_end(ClosureInputTranslucent cl_in,
                                      ClosureEvalTranslucent cl_eval,
                                      ClosureEvalCommon cl_common,
                                      inout ClosureOutputTranslucent cl_out)
{
  /* If not enough light has been accumulated from probes, use the world specular cubemap
   * to fill the remaining energy needed. */
  if (cl_common.diffuse_accum > 0.0) {
    vec3 probe_radiance = probe_evaluate_world_diff(cl_in.N);
    cl_out.radiance += cl_common.diffuse_accum * probe_radiance;
  }
}

void closure_Translucent_eval_end(ClosureInputTranslucent cl_in,
                                  ClosureEvalTranslucent cl_eval,
                                  ClosureEvalCommon cl_common,
                                  inout ClosureOutputTranslucent cl_out)
{
#if defined(DEPTH_SHADER) || defined(WORLD_BACKGROUND)
  /* This makes shader resources become unused and avoid issues with samplers. (see T59747) */
  cl_out.radiance = vec3(0.0);
  return;
#endif
}

/** \} */
