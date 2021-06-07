
/**
 * Simple passthrough shader. Outputs depth without ammendment.
 */

#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_bsdf_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_nodetree_eval_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_sampling_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shader_shared.hh)
#pragma BLENDER_REQUIRE(eevee_surface_lib.glsl)

layout(std140) uniform sampling_block
{
  SamplingData sampling;
};

utility_tx_sample_define_stub(utility_tx);

void main(void)
{
  /* No color output, only depth (line below is implicit). */
  /* gl_FragDepth = gl_FragCoord.z; */
}