
/**
 * Custom vertex shader for rendering the lookdev overlay (reference material spheres).
 * The input mesh is a sphere. The output is a flattened version that will render at depth 0.
 **/

#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_surface_lib.glsl)

in vec3 pos;
in vec3 nor;

/* TODO(fclem) remove use of macro. use interface block instead. */
RESOURCE_ID_VARYING

void main(void)
{
  interp.P = pos;
  interp.N = nor;
  interp.barycentric_coords = vec2(0.0);
  interp.barycentric_dists = vec3(0.0);

  PASS_RESOURCE_ID

  /* Camera transform is passed via the model matrix. */
  gl_Position.xyz = transform_direction(ModelMatrix, interp.P);
  /* Apply packed bias & scale. */
  gl_Position.xy *= ModelMatrix[3].xy;
  gl_Position.xy += ModelMatrix[3].zw;

  /* Override depth. */
  gl_Position.z = -1.0;
  gl_Position.w = 1.0;
}
