
#pragma BLENDER_REQUIRE(common_attribute_lib.glsl)
#pragma BLENDER_REQUIRE(common_hair_lib.glsl)
#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_nodetree_eval_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_surface_lib.glsl)

/* TODO(fclem) remove use of macro. use interface block instead. */
RESOURCE_ID_VARYING

/* Globals to feed the load functions. */
vec3 T;

void main(void)
{
  bool is_persp = (ProjectionMatrix[3][3] == 0.0);
  hair_get_pos_tan_binor_time(is_persp,
                              ModelMatrixInverse,
                              ViewMatrixInverse[3].xyz,
                              ViewMatrixInverse[2].xyz,
                              interp.P,
                              T,
                              interp.hair_binormal,
                              interp.hair_time,
                              interp.hair_thickness,
                              interp.hair_time_width);

  interp.N = cross(T, interp.hair_binormal);
  interp.hair_strand_id = hair_get_strand_id();
  interp.barycentric_coords = hair_get_barycentric();

  PASS_RESOURCE_ID
  attrib_load();

  g_data = init_globals();
  interp.P += nodetree_displacement();

  gl_Position = point_world_to_ndc(interp.P);
}

#ifdef OBINFO_LIB
vec3 attr_load_orco(samplerBuffer cd_buf)
{
  vec3 P = hair_get_strand_pos();
  vec3 lP = transform_point(ModelMatrixInverse, P);
  return OrcoTexCoFactors[0].xyz + lP * OrcoTexCoFactors[1].xyz;
}
#endif

vec4 attr_load_tangent(samplerBuffer cd_buf)
{
  /* Not supported. */
  return vec4(0.0, 0.0, 0.0, 1.0);
}

vec3 attr_load_uv(samplerBuffer cd_buf)
{
  return texelFetch(cd_buf, interp.hair_strand_id).rgb;
}

vec4 attr_load_color(samplerBuffer cd_buf)
{
  return texelFetch(cd_buf, interp.hair_strand_id).rgba;
}

vec4 attr_load_vec4(samplerBuffer cd_buf)
{
  return texelFetch(cd_buf, interp.hair_strand_id).rgba;
}

vec3 attr_load_vec3(samplerBuffer cd_buf)
{
  return texelFetch(cd_buf, interp.hair_strand_id).rgb;
}

vec2 attr_load_vec2(samplerBuffer cd_buf)
{
  return texelFetch(cd_buf, interp.hair_strand_id).rg;
}
