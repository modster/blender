#pragma BLENDER_REQUIRE(lights_lib.glsl)
#pragma BLENDER_REQUIRE(lightprobe_lib.glsl)
#pragma BLENDER_REQUIRE(ambient_occlusion_lib.glsl)

struct ClosureInputDiffuse {
  vec3 N;      /** Shading normal. */
  vec3 albedo; /** Used for multibounce GTAO approximation. Not applied to final radiance. */
};

#define CLOSURE_INPUT_Diffuse_DEFAULT ClosureInputDiffuse(vec3(0.0), vec3(0.0))

struct ClosureEvalDiffuse {
  float dummy;
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
  cl_eval.dummy = 0.0;
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
      grid.data, cl_common.P, cl_common.bent_normal, grid.local_pos);
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
    vec3 probe_radiance = probe_evaluate_world_diff(cl_common.bent_normal);
    cl_out.radiance += cl_common.diffuse_accum * probe_radiance;
  }
  /* Apply occlusion on radiance before the light loop. */
  cl_out.radiance *= gtao_multibounce(cl_common.occlusion, cl_in.albedo);
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
