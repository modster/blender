
#pragma BLENDER_REQUIRE(eevee_bsdf_stubs_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_nodetree_eval_lib.glsl)

/**
 * Simple passthrough shader. Outputs depth without ammendment.
 */

void main(void)
{
  /* No color output, only depth (line below is implicit). */
  /* gl_FragDepth = gl_FragCoord.z; */
}
