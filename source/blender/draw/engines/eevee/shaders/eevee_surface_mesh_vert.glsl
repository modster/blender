
#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(common_attribute_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_nodetree_eval_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_surface_lib.glsl)

in vec3 pos;
in vec3 nor;

void main(void)
{
  interp.P = point_object_to_world(pos);
  interp.N = normal_object_to_world(nor);
  interp.barycentric_coords = vec2(0.0);
  interp.barycentric_dists = vec3(0.0);

  attrib_load();

  g_data = init_globals();

  interp.P += nodetree_displacement();

  gl_Position = point_world_to_ndc(interp.P);
}

#ifdef OBINFO_LIB
vec3 attr_load_orco(vec4 orco)
{
  /* We know when there is no orco layer when orco.w is 1.0 because it uses the generic vertex
   * attrib (which is [0,0,0,1]). */
  if (orco.w == 0.0) {
    return orco.xyz * 0.5 + 0.5;
  }
  else {
    /* If the object does not have any deformation, the orco layer calculation is done on the fly
     * using the orco_madd factors. */
    return OrcoTexCoFactors[0].xyz + pos * OrcoTexCoFactors[1].xyz;
  }
}
#endif

vec4 attr_load_tangent(vec4 tangent)
{
  tangent.xyz = safe_normalize(normal_object_to_world(tangent.xyz));
  return tangent;
}

/* Simple passthrough. */
vec4 attr_load_vec4(vec4 attr)
{
  return attr;
}
vec3 attr_load_vec3(vec3 attr)
{
  return attr;
}
vec2 attr_load_vec2(vec2 attr)
{
  return attr;
}