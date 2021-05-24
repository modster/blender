
#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(common_attribute_lib.glsl)
#pragma BLENDER_REQUIRE(common_gpencil_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_surface_lib.glsl)

in ivec4 ma, ma1, ma2, ma3;
in vec4 pos, pos1, pos2, pos3, uv1, uv2, col1, col2, fcol1;

/* Globals to feed the load functions */
vec2 uvs;
vec4 color;

void main(void)
{
  vec4 sspos;
  vec2 aspect;
  vec2 thickness;
  vec4 viewport_size = vec4(1500.0, 1500.0, 1.0 / 1500.0, 1.0 / 1500.0);

  gl_Position = gpencil_vertex(ma,
                               ma1,
                               ma2,
                               ma3,
                               pos,
                               pos1,
                               pos2,
                               pos3,
                               uv1,
                               uv2,
                               col1,
                               col2,
                               fcol1,
                               viewport_size,
                               interp.P,
                               color,
                               uvs,
                               sspos,
                               aspect,
                               thickness);

  interp.N = normal_object_to_world(vec3(0.0, 1.0, 0.0)); /* TODO */
  interp.barycentric_coords = vec2(0.0);
  interp.barycentric_dists = vec3(0.0);

  attrib_load();
}

#ifdef OBINFO_LIB
vec3 attr_load_orco(vec4 orco)
{
  vec3 lP = point_world_to_object(interp.P);
  return OrcoTexCoFactors[0].xyz + lP * OrcoTexCoFactors[1].xyz;
}
#endif

vec4 attr_load_tangent(vec4 tangent)
{
  /* TODO */
  return vec4(0.0, 0.0, 0.0, 1.0);
}

/* Only have one uv and one color attribute layer. */
vec3 attr_load_uv(vec3 dummy)
{
  return vec3(uvs, 0.0);
}
vec4 attr_load_color(vec4 dummy)
{
  return color;
}

/* Not supported. */
vec4 attr_load_vec4(vec4 attr)
{
  return vec4(0.0);
}
vec3 attr_load_vec3(vec3 attr)
{
  return vec3(0.0);
}
vec2 attr_load_vec2(vec2 attr)
{
  return vec2(0.0);
}