
/**
 * Fullscreen pass that filter previous mipmap level using a 1 bilinear tap.
 * This uses layered rendering to filter all cubeface / layers in one drawcall.
 */

#pragma BLENDER_REQUIRE(eevee_lightprobe_filter_lib.glsl)

#ifdef CUBEMAP
uniform samplerCube input_tx;
#else
uniform sampler2DArray input_tx;
#endif

layout(location = 0) out vec4 out_color;

void main(void)
{
  out_color = textureLod(input_tx, interp.coord, 0.0);
}
