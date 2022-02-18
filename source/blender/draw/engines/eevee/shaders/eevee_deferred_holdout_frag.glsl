
/**
 * Save radiance from main pass to subtract to final render.
 *
 * This way all screen space effects (SSS, SSR) are not altered by the presence of the holdout.
 **/

#pragma BLENDER_REQUIRE(eevee_gbuffer_lib.glsl)

void main(void)
{
  vec3 combined_radiance = texture(combined_tx, uvcoordsvar.xy).rgb;

  ClosureTransparency transparency_data = gbuffer_load_transparency_data(transparency_data_tx,
                                                                         uvcoordsvar.xy);

  out_holdout = combined_radiance * transparency_data.holdout;
}
