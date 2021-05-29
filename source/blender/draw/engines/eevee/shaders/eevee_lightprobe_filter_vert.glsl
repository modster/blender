
/**
 * Fullscreen pass that outputs one triangle to a specific layer.
 * This uses layered rendering to filter all cubeface / layers in one drawcall.
 */

#pragma BLENDER_REQUIRE(eevee_lightprobe_filter_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shader_shared.hh)

layout(std140) uniform filter_block
{
  LightProbeFilterData probe;
};

void main(void)
{
  /* Fullscreen triangle. */
  int v = gl_VertexID % 3;
  interp.coord.x = float((v & 1) << 2);
  interp.coord.y = float((v & 2) << 1);
  gl_Position = vec4(interp.coord.xy * 2.0 - 1.0, 1.0, 1.0);

  int cube_face = gl_VertexID / 3;
  interp.layer = probe.target_layer + cube_face;
  interp.coord.z = float(interp.layer);

#ifdef CUBEMAP
  switch (cube_face) {
    case 0:
      interp.coord = gl_Position.zyx * vec3(1, -1, -1);
      break;
    case 1:
      interp.coord = gl_Position.zyx * vec3(-1, -1, 1);
      break;
    case 2:
      interp.coord = gl_Position.xzy * vec3(1, 1, 1);
      break;
    case 3:
      interp.coord = gl_Position.xzy * vec3(1, -1, -1);
      break;
    case 4:
      interp.coord = gl_Position.xyz * vec3(1, -1, 1);
      break;
    default:
      interp.coord = gl_Position.xyz * vec3(-1, -1, -1);
      break;
  }
#endif
}
