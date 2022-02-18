
#pragma BLENDER_REQUIRE(eevee_film_lib.glsl)

void main(void)
{
  vec2 uv = uvcoordsvar.xy;

  vec4 color = textureLod(data_tx, uv, 0.0);
  float weight = textureLod(weight_tx, uv, 0.0).r;

  gl_FragDepth = film_data_decode(film, color, weight).r;
}
