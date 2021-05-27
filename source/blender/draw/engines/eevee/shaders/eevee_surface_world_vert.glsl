
#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_nodetree_eval_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_surface_lib.glsl)

/* TODO(fclem) remove use of macro. use interface block instead. */
RESOURCE_ID_VARYING

void main(void)
{
  /* Fullscreen triangle. */
  int v = gl_VertexID % 3;
  float x = float((v & 1) << 2) - 1.0;
  float y = float((v & 2) << 1) - 1.0;
  gl_Position = vec4(x, y, 1.0, 1.0);

  /* Pass view position to keep accuracy. */
  interp.P = project_point(ProjectionMatrixInverse, gl_Position.xyz);
  interp.N = vec3(1);
  /* Unsupported. */
  interp.barycentric_coords = vec2(0.0);
  interp.barycentric_dists = vec3(0.0);

  /* Used to pass the correct model matrix. */
  PASS_RESOURCE_ID
}
