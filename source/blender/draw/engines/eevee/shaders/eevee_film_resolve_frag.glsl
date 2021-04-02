
#pragma BLENDER_REQUIRE(eevee_film_lib.glsl)

layout(std140) uniform film_block
{
  FilmData film;
};

uniform sampler2D data_tx;
uniform sampler2D weight_tx;

in vec4 uvcoordsvar;

layout(location = 0) out vec4 out_color;

void main(void)
{
  vec2 uv = uvcoordsvar.xy;

  vec4 color = textureLod(data_tx, uv, 0.0);
  float weight = textureLod(weight_tx, uv, 0.0).r;

  out_color = film_data_decode(film, color, weight);
  gl_FragDepth = out_color.r;
}