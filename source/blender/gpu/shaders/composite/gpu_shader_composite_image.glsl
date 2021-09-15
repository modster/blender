
void node_composite_image(sampler2D tex, out vec4 result)
{
  /* TODO(fclem) correct positionning. */
  result = texture(tex, g_data.uv_texco);
}

void node_composite_image_empty(out vec4 result)
{
  result = vec4(0.0);
}

void node_composite_rlayers(sampler2D rlayer, out vec4 result)
{
  result = texture(rlayer, g_data.uv_render_layer);
}
