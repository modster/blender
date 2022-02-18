
#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(gpu_shader_codegen_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_nodetree_eval_lib.glsl)

/* To be fed the result of hair_get_barycentric from vertex shader. */
vec2 hair_resolve_barycentric(vec2 vert_barycentric)
{
  if (fract(vert_barycentric.y) != 0.0) {
    return vec2(vert_barycentric.x, 0.0);
  }
  else {
    return vec2(1.0 - vert_barycentric.x, 0.0);
  }
}

#if defined(GPU_FRAGMENT_SHADER) || defined(GPU_VERTEX_SHADER)
GlobalData init_globals(void)
{
  GlobalData surf;
  surf.P = interp.P;
  surf.N = normalize(interp.N);
#  ifndef MAT_GEOM_GPENCIL
  surf.N = (FrontFacing) ? surf.N : -surf.N;
#  endif
#  ifdef GPU_FRAGMENT_SHADER
  surf.Ng = safe_normalize(cross(dFdx(surf.P), dFdy(surf.P)));
#  else
  surf.Ng = surf.N;
#  endif

#  ifdef MAT_GEOM_HAIR
  /* Shade as a cylinder. */
  float cos_theta = interp.hair_time_width / interp.hair_thickness;
  float sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
  surf.N = normalize(interp.N * sin_theta + interp.hair_binormal * cos_theta);
#  endif

#  ifdef MAT_GEOM_HAIR
  surf.is_strand = true;
  surf.hair_time = interp.hair_time;
  surf.hair_thickness = interp.hair_thickness;
  surf.hair_strand_id = interp.hair_strand_id;
  surf.barycentric_coords = hair_resolve_barycentric(interp.barycentric_coords);
#  else
  surf.is_strand = false;
  surf.hair_time = 0.0;
  surf.hair_thickness = 0.0;
  surf.hair_strand_id = 0;
  surf.barycentric_coords = interp.barycentric_coords;
#  endif
  surf.barycentric_dists = interp.barycentric_dists;
  surf.ray_type = RAY_TYPE_CAMERA;
  surf.ray_depth = 0.0;
  surf.ray_length = distance(surf.P, cameraPos);
  surf.closure_rand = 0.5;
  return surf;
}
#endif
