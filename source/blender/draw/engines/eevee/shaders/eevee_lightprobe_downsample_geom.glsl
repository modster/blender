
/**
 * Fullscreen pass that filter previous mipmap level using a 1 bilinear tap.
 * This uses layered rendering to filter all cubeface / layers in one drawcall.
 */

/* TODO(fclem) Use vendor extensions to bypass geometry shader. */

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

in downsampleInterface
{
  vec3 coord;
  flat int layer;
}
interp_in[];

out downsampleInterface
{
  vec3 coord;
  flat int layer;
}
interp_out;

void main()
{
  gl_Layer = interp_in[0].layer;

  interp_out.coord = interp_in[0].coord;
  gl_Position = gl_in[0].gl_Position;
  EmitVertex();

  interp_out.coord = interp_in[1].coord;
  gl_Position = gl_in[1].gl_Position;
  EmitVertex();

  interp_out.coord = interp_in[2].coord;
  gl_Position = gl_in[2].gl_Position;
  EmitVertex();

  EndPrimitive();
}
