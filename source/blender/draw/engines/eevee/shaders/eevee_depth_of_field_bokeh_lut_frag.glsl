
/**
 * Bokeh Look Up Table: This outputs a radius multiplier to shape the sampling in gather pass or
 * the scatter sprite appearance. This is only used if bokeh shape is either anamorphic or is not
 * a perfect circle.
 * We correct samples spacing for polygonal bokeh shapes. However, we do not for anamorphic bokeh
 * as it is way more complex and expensive to do.
 **/

#pragma BLENDER_REQUIRE(eevee_depth_of_field_lib.glsl)

layout(std140) uniform dof_block
{
  DepthOfFieldData dof;
};

in vec4 uvcoordsvar;

layout(location = 0) out vec2 out_gather_lut;
layout(location = 1) out float out_scatter_Lut;
layout(location = 2) out float outResolveLut;

void main()
{
  /* Center uv in range [-1..1]. */
  vec2 uv = uvcoordsvar.xy * 2.0 - 1.0;

  float radius = length(uv);

  vec2 texel = floor(gl_FragCoord.xy) - float(dof_max_slight_focus_radius);

  if (dof.bokeh_blades > 0.0) {
    /* NOTE: atan(y,x) has output range [-M_PI..M_PI], so add 2pi to avoid negative angles. */
    float theta = atan(uv.y, uv.x) + M_2PI;
    float r = length(uv);

    radius /= circle_to_polygon_radius(dof.bokeh_blades, theta - dof.bokeh_rotation);

    float theta_new = circle_to_polygon_angle(dof.bokeh_blades, theta);
    float r_new = circle_to_polygon_radius(dof.bokeh_blades, theta_new);

    theta_new -= dof.bokeh_rotation;

    uv = r_new * vec2(-cos(theta_new), sin(theta_new));

    {
      /* Slight focus distance */
      texel *= dof.bokeh_anisotropic_scale_inv;
      float theta = atan(texel.y, -texel.x) + M_2PI;
      texel /= circle_to_polygon_radius(dof.bokeh_blades, theta + dof.bokeh_rotation);
    }
  }
  else {
    uv *= safe_rcp(length(uv));
  }

  /* For gather store the normalized UV. */
  out_gather_lut = uv;
  /* For scatter store distance. */
  out_scatter_Lut = radius;
  /* For slight focus gather store pixel perfect distance. */
  outResolveLut = length(texel);
}
