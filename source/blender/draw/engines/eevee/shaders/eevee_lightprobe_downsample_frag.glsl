
/**
 * Fullscreen pass that filter previous mipmap level using a 1 bilinear tap.
 * This uses layered rendering to filter all cubeface / layers in one drawcall.
 */

#ifdef CUBEMAP
uniform samplerCube input_tx;
#else
uniform sampler2DArray input_tx;
#endif

in downsampleInterface
{
  vec3 coord;
  flat int layer;
};

out vec4 out_color;

void main(void)
{
  out_color = clamp(textureLod(input_tx, coord, 0.0), 0.0, 1e20);
}
