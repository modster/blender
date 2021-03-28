
#pragma BLENDER_REQUIRE(common_math_lib.glsl)

uniform sampler2D dataTexture;
uniform sampler2D weightTexture;

in vec4 uvcoordsvar;

layout(location = 0) out vec4 outColor;

vec4 accumulator_colorspace_decode(vec4 color)
{
  color.rgb = exp2(color.rgb) - 1.0;
  return color;
}

void main(void)
{
  vec2 uv = uvcoordsvar.xy;

  vec4 color = textureLod(dataTexture, uv, 0.0);
  float weight = textureLod(weightTexture, uv, 0.0).r;

  outColor = accumulator_colorspace_decode(color * safe_rcp(weight));
}