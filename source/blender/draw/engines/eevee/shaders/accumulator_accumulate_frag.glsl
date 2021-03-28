
uniform mat4 dstViewProjectionMatrix;
uniform mat4 dstViewProjectionInverse;
uniform mat4 srcViewProjectionMatrix;
uniform mat4 srcViewProjectionInverse;
uniform ivec2 targetSize;

uniform sampler2D inputTexture;

in vec4 uvcoordsvar;

layout(location = 0) out vec4 outData;
layout(location = 1) out float outWeight;

vec4 accumulator_colorspace_encode(vec4 color)
{
  color.rgb = log2(1.0 + color.rgb);
  return color;
}

#if 0
vec2 accumulator_view_projection_distort(vec2 uv)
{
  /* TODO(fclem) Panoramic projection. */
  return view_from_uv(dstViewProjectionInverse, uv_dst);
}

vec2 accumulator_view_projection_undistort(vec2 uv)
{
  /* TODO(fclem) Panoramic projection. */
  return uv_from_view(dstViewProjectionMatrix, V);
}

void main(void)
{
  vec2 uv_dst = uvcoordsvar.xy;
  vec2 texel_center_dst = uv_dst * vec2(targetSize);

  vec3 V = accumulator_view_projection_distort(uv_dst);
  vec2 uv_src = uv_from_view(srcViewProjectionMatrix, V);

  vec2 input_size = vec2(textureSize(inputTexture, 0));
  vec2 input_size_inv = 1.0 / input_size;
  vec2 texel = uv_src * input_size;
  vec2 texel_low = floor(texel);
  vec2 texel_high = texel_low + 1.0;

  /* Compute each 4 bilinear tap location. */
  vec2 samp_uv[4];
  samp_uv[0] = texel_low * input_size_inv;
  samp_uv[1] = vec2(texel_low.x, texel_high.y) * input_size_inv;
  samp_uv[2] = vec2(texel_high.x, texel_low.y) * input_size_inv;
  samp_uv[3] = texel_high * input_size_inv;

  outData = vec4(0.0);
  outWeight = 0.0;

  for (int i = 0; i < 4; i++) {
    /* Reproject sample location in dst space. */
    vec3 V = view_from_uv(srcViewProjectionInverse, samp_uv[i]);
    vec2 uv_dst = accumulator_view_projection_undistort(V);
    vec2 texel_dst = uv_dst * vec2(targetSize);

    vec2 offset = texel_dst - texel_center_dst;
    float weight = 0.0;  // blackmann_harris_weight(offset, filter_size);
    /* Add a small linear weight to avoid really small filter having 0 weight. */
    weight += (filter_size - length(offset)) * 0.0001;

    vec4 data = textureLod(inputTexture, samp_uv[i], 0.0);

    outData += accumulator_colorspace_encode(data) * weight;
    outWeight += weight;
  }
}
#endif

void main(void)
{
  vec2 uv = uvcoordsvar.xy;
  vec4 data = textureLod(inputTexture, uv, 0.0);
  float weight = 1.0;
  outData += accumulator_colorspace_encode(data) * weight;
  outWeight += weight;
}