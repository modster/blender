#define CMP_NODE_MASKTYPE_ADD 0.0
#define CMP_NODE_MASKTYPE_SUBTRACT 1.0
#define CMP_NODE_MASKTYPE_MULTIPLY 2.0
#define CMP_NODE_MASKTYPE_NOT 3.0

void node_composite_ellipse_mask(float in_mask,
                                 float value,
                                 const float mask_type,
                                 float x_location,
                                 float y_location,
                                 float half_width,
                                 float half_height,
                                 float cos_angle,
                                 float sin_angle,
                                 out float out_mask)
{
  vec2 uv = g_data.uv_render_layer.xy;
  uv -= vec2(x_location, y_location);
  uv.y *= ViewportSize.y / ViewportSize.x;
  uv = mat2(cos_angle, -sin_angle, sin_angle, cos_angle) * uv;
  bool is_inside = length(uv / vec2(half_width, half_height)) < 1.0;

  if (mask_type == CMP_NODE_MASKTYPE_ADD) {
    out_mask = is_inside ? max(in_mask, value) : in_mask;
  }
  else if (mask_type == CMP_NODE_MASKTYPE_SUBTRACT) {
    out_mask = is_inside ? clamp(in_mask - value, 0.0, 1.0) : in_mask;
  }
  else if (mask_type == CMP_NODE_MASKTYPE_MULTIPLY) {
    out_mask = is_inside ? in_mask * value : 0.0;
  }
  else if (mask_type == CMP_NODE_MASKTYPE_NOT) {
    out_mask = is_inside ? (in_mask > 0.0 ? 0.0 : value) : in_mask;
  }
}

#undef CMP_NODE_MASKTYPE_ADD
#undef CMP_NODE_MASKTYPE_SUBTRACT
#undef CMP_NODE_MASKTYPE_MULTIPLY
#undef CMP_NODE_MASKTYPE_NOT
