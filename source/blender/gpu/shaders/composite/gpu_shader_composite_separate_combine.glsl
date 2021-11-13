/* ** Combine/Separate RGBA ** */

void node_composite_combine_rgba(float r, float g, float b, float a, out vec4 color)
{
  color = vec4(r, g, b, a);
}

void node_composite_separate_rgba(vec4 color, out float r, out float g, out float b, out float a)
{
  r = color.r;
  g = color.g;
  b = color.b;
  a = color.a;
}

/* ** Combine/Separate HSVA ** */

void node_composite_combine_hsva(float h, float s, float v, float a, out vec4 color)
{
  hsv_to_rgb(vec4(h, s, v, a), color);
}

void node_composite_separate_hsva(vec4 color, out float h, out float s, out float v, out float a)
{
  vec4 hsva;
  rgb_to_hsv(color, hsva);
  h = hsva.x;
  s = hsva.y;
  v = hsva.z;
  a = hsva.a;
}

/* ** Combine/Separate YCCA ** */

void node_composite_combine_ycca_itu_601(float y, float cb, float cr, float a, out vec4 color)
{
  ycca_to_rgba_itu_601(vec4(y, cb, cr, a), color);
}

void node_composite_combine_ycca_itu_709(float y, float cb, float cr, float a, out vec4 color)
{
  ycca_to_rgba_itu_709(vec4(y, cb, cr, a), color);
}

void node_composite_combine_ycca_jpeg(float y, float cb, float cr, float a, out vec4 color)
{
  ycca_to_rgba_jpeg(vec4(y, cb, cr, a), color);
}

void node_composite_separate_ycca_itu_601(
    vec4 color, out float y, out float cb, out float cr, out float a)
{
  vec4 ycca;
  rgba_to_ycca_itu_601(color, ycca);
  y = ycca.x;
  cb = ycca.y;
  cr = ycca.z;
  a = ycca.a;
}

void node_composite_separate_ycca_itu_709(
    vec4 color, out float y, out float cb, out float cr, out float a)
{
  vec4 ycca;
  rgba_to_ycca_itu_709(color, ycca);
  y = ycca.x;
  cb = ycca.y;
  cr = ycca.z;
  a = ycca.a;
}

void node_composite_separate_ycca_jpeg(
    vec4 color, out float y, out float cb, out float cr, out float a)
{
  vec4 ycca;
  rgba_to_ycca_jpeg(color, ycca);
  y = ycca.x;
  cb = ycca.y;
  cr = ycca.z;
  a = ycca.a;
}

/* ** Combine/Separate YUVA ** */

void node_composite_combine_yuva_itu_709(float y, float u, float v, float a, out vec4 color)
{
  yuva_to_rgba_itu_709(vec4(y, u, v, a), color);
}

void node_composite_separate_yuva_itu_709(
    vec4 color, out float y, out float u, out float v, out float a)
{
  vec4 yuva;
  rgba_to_yuva_itu_709(color, yuva);
  y = yuva.x;
  u = yuva.y;
  v = yuva.z;
  a = yuva.a;
}
