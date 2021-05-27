
#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(common_attribute_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_surface_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_bsdf_stubs_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_nodetree_eval_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shader_shared.hh)

layout(location = 0) out vec4 out_background;

void main(void)
{
  g_data = init_globals();
  /* View position is passed to keep accuracy. */
  g_data.N = normal_view_to_world(viewCameraVec(interp.P));
  g_data.Ng = g_data.N;
  g_data.P = -g_data.N + cameraPos;

  attrib_load();

  nodetree_surface();

  out_background.rgb = safe_color(g_emission_data.emission);
  out_background.a = saturate(1.0 - avg(g_transparency_data.transmittance));
}

vec3 attr_load_orco(vec4 orco)
{
  return -g_data.N;
}

/* Unsupported. */
vec4 attr_load_tangent(vec4 tangent)
{
  return vec4(0);
}
vec4 attr_load_vec4(vec4 attr)
{
  return vec4(0);
}
vec3 attr_load_vec3(vec3 attr)
{
  return vec3(0);
}
vec2 attr_load_vec2(vec2 attr)
{
  return vec2(0);
}
vec4 attr_load_color(vec4 attr)
{
  return vec4(0);
}
vec3 attr_load_uv(vec3 attr)
{
  return vec3(0);
}
