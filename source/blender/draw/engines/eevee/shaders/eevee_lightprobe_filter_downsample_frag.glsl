
/**
 * Fullscreen pass that filter previous mipmap level using a 1 bilinear tap.
 * This uses layered rendering to filter all cubeface / layers in one drawcall.
 */

void main(void)
{
  out_color = textureLod(input_tx, interp.coord, 0.0);
}
