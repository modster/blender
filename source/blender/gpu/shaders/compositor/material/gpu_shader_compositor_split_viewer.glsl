void node_composite_split_viewer_x(vec4 first_color,
                                   vec4 second_color,
                                   float split_factor,
                                   out vec4 result)
{
  result = g_data.uv_render_layer.x > split_factor ? first_color : second_color;
}

void node_composite_split_viewer_y(vec4 first_color,
                                   vec4 second_color,
                                   float split_factor,
                                   out vec4 result)
{
  result = g_data.uv_render_layer.y > split_factor ? first_color : second_color;
}
