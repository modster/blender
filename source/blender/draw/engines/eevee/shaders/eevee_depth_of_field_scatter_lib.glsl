
#pragma BLENDER_REQUIRE(eevee_depth_of_field_lib.glsl)

IN_OUT DofScatterStageInterface
{
  /** Colors, weights, and Circle of confusion radii for the 4 pixels to scatter. */
  flat vec4 color1;
  flat vec4 color2;
  flat vec4 color3;
  flat vec4 color4;
  flat vec4 weights;
  flat vec4 cocs;
  /** Sprite center position. In pixels. */
  flat vec2 spritepos;
  /* MaxCoC */
  flat float spritesize;
};
