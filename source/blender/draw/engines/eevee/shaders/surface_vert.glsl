
#pragma BLENDER_REQUIRE(common_hair_lib.glsl)
#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(surface_lib.glsl)
#pragma BLENDER_REQUIRE(gpencil_common_lib.glsl)

#if !defined(HAIR_SHADER) && !defined(GPENCIL_SHADER)
in vec3 nor;
#endif

RESOURCE_ID_VARYING

void main()
{
  GPU_INTEL_VERTEX_SHADER_WORKAROUND

  PASS_RESOURCE_ID

#ifdef HAIR_SHADER
  hairStrandID = hair_get_strand_id();
  vec3 pos, binor;
  hair_get_pos_tan_binor_time((ProjectionMatrix[3][3] == 0.0),
                              ModelMatrixInverse,
                              ViewMatrixInverse[3].xyz,
                              ViewMatrixInverse[2].xyz,
                              pos,
                              hairTangent,
                              binor,
                              hairTime,
                              hairThickness,
                              hairThickTime);
  worldNormal = cross(hairTangent, binor);
  vec3 world_pos = pos;
#elif defined(GPENCIL_SHADER)
  /* TODO */
  vec3 pos = vec3(0.0);
  vec3 nor = vec3(1.0);
  vec3 world_pos = pos.xyz;
#else
  vec3 world_pos = point_object_to_world(pos.xyz);
#endif

  gl_Position = point_world_to_ndc(world_pos);

#if defined(GPENCIL_SHADER)
  gpencil_vertex();
#endif

  /* Used for planar reflections */
  gl_ClipDistance[0] = dot(vec4(world_pos, 1.0), clipPlanes[0]);

#ifdef MESH_SHADER
  worldPosition = world_pos;
  viewPosition = point_world_to_view(worldPosition);

#  ifndef HAIR_SHADER
  worldNormal = normalize(normal_object_to_world(nor));
#  endif

  /* No need to normalize since this is just a rotation. */
  viewNormal = normal_world_to_view(worldNormal);
#  ifdef USE_ATTR
#    ifdef HAIR_SHADER
  pos = hair_get_strand_pos();
#    endif
  pass_attr(pos.xyz, NormalMatrix, ModelMatrixInverse);
#  endif
#endif
}
