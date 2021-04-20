uniform sampler2D maskBuf;
in vec4 uvcoordsvar;

/* Blend mode is multiply. */
layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 fragRevealage;

void main()
{
  float mask = textureLod(maskBuf, uvcoordsvar.xy, 0).r;
  fragRevealage = fragColor = vec4(1.0 - mask);
}
