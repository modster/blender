
/**
 * Fullscreen pass that filter previous mipmap level using a 1 bilinear tap.
 * This uses layered rendering to filter all cubeface / layers in one drawcall.
 */

#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(common_math_lib.glsl)

out downsampleInterface
{
  vec3 coord;
  flat int layer;
};

void main(void)
{
  /* Fullscreen triangle. */
  int v = gl_VertexID % 3;
  float x = float((v & 1) << 2) - 1.0;
  float y = float((v & 2) << 1) - 1.0;
  gl_Position = vec4(x, y, 1.0, 1.0);

  layer = gl_VertexID / 3;

#ifdef CUBEMAP
  switch (layer) {
    case 0:
      coord = gl_Position.zyx * vec3(1, -1, -1);
      break;
    case 1:
      coord = gl_Position.zyx * vec3(-1, -1, 1);
      break;
    case 2:
      coord = gl_Position.xzy * vec3(1, 1, 1);
      break;
    case 3:
      coord = gl_Position.xzy * vec3(1, -1, -1);
      break;
    case 4:
      coord = gl_Position.xyz * vec3(1, -1, 1);
      break;
    default:
      coord = gl_Position.xyz * vec3(-1, -1, -1);
      break;
  }
#else
  coord.xy = gl_Position * 0.5 + 0.5;
  coord.z = float(layer);
#endif
}
