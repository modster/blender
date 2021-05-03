
/**
 * Renders volume objects with no surfaces.
 *
 * The vertex shader outputs geometry at nearest depth.
 **/

#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_volume_lib.glsl)

in vec3 pos;

void main(void)
{
  /* TODO(fclem) Make the quad cover only the bounding box. */

  // int v = gl_VertexID % 4;
  // float x = -1.0 + float((v & 1) << 2);
  // float y = -1.0 + float((v & 2) << 1);
  // gl_Position = vec4(x, y, 1.0, 1.0);

  // vec3 aabb_min,;

  // uvcoordsvar = vec4((gl_Position.xy + 1.0) * 0.5, 0.0, 0.0);

  // interp.P_start = point_ndc_to_world(pos);
  // interp.P_end = point_ndc_to_world(pos);

  interp.P_start = point_object_to_world(pos);
  interp.P_end = interp.P_start;

  gl_Position = point_world_to_ndc(interp.P_start);
}