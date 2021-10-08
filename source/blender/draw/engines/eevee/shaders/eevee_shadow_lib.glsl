
#pragma BLENDER_REQUIRE(eevee_shader_shared.hh)

/* ---------------------------------------------------------------------- */
/** \name Shadow Sampling Functions
 * \{ */

/* Turns local light coordinate into shadow region index. Matches eShadowCubeFace order. */
int shadow_punctual_face_index_get(vec3 lL)
{
  vec3 aP = abs(lL);
  if (all(greaterThan(aP.xx, aP.yz))) {
    return (lL.x > 0.0) ? 1 : 2;
  }
  else if (all(greaterThan(aP.yy, aP.xz))) {
    return (lL.y > 0.0) ? 3 : 4;
  }
  else {
    return (lL.z > 0.0) ? 5 : 0;
  }
}

/* Transform vector to face local coordinate. */
vec3 shadow_punctual_local_position_to_face_local(int face_id, vec3 lL)
{
  switch (face_id) {
    case 1:
      return vec3(-lL.y, lL.z, -lL.x);
    case 2:
      return vec3(lL.y, lL.z, lL.x);
    case 3:
      return vec3(lL.x, lL.z, -lL.y);
    case 4:
      return vec3(-lL.x, lL.z, lL.y);
    case 5:
      return vec3(lL.x, -lL.y, -lL.z);
    default:
      return lL;
  }
}

vec3 shadow_punctual_coordinates_get(ShadowPunctualData data, vec3 lL)
{
  lL -= data.shadow_offset;
  int face_id = shadow_punctual_face_index_get(lL);
  lL = shadow_punctual_local_position_to_face_local(face_id, lL);
  lL *= min(0.0, lL.z + data.shadow_bias) / lL.z;

  vec3 shadow_co = project_point(data.shadow_mat, lL);
  /* Add one more offset if not omni because region_offset is half a face (in this case) and
   * only the first face is full sized. */
  float offset_fac = float(face_id > 0 && !data.is_omni) + float(face_id);
  shadow_co.y += offset_fac * data.region_offset;
  return shadow_co;
}

/* Returns world distance delta from light between shading point (lL) and shadow depth. */
float shadow_punctual_depth_delta(ShadowPunctualData data, vec3 lL, float depth)
{
  lL -= data.shadow_offset;
  /* Revert the constant bias from shadow rendering. (Tweaked for 16bit shadowmaps) */
  const float depth_bias = 3.1e-5;
  depth = saturate(depth - depth_bias);
  depth = linear_depth(true, depth, data.shadow_far, data.shadow_near);
  depth *= length(lL / max_v3(abs(lL)));
  return length(lL) - depth;
}

/** \} */
