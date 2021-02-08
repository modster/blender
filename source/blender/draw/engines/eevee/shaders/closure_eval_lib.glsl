
#pragma BLENDER_REQUIRE(common_utiltex_lib.glsl)
#pragma BLENDER_REQUIRE(lights_lib.glsl)
#pragma BLENDER_REQUIRE(lightprobe_lib.glsl)

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
