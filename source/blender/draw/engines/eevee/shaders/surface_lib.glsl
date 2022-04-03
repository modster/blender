/** This describe the entire interface of the shader. */

#define SURFACE_INTERFACE \
  vec3 worldPosition; \
  vec3 viewPosition; \
  vec3 worldNormal; \
  vec3 viewNormal;

#ifndef IN_OUT
#  if defined(GPU_VERTEX_SHADER)
#    define IN_OUT out
#  elif defined(GPU_FRAGMENT_SHADER)
#    define IN_OUT in
#  endif
#endif

#ifndef EEVEE_GENERATED_INTERFACE
#  if defined(STEP_RESOLVE) || defined(STEP_RAYTRACE)
/* SSR will set these global variables itself.
 * Also make false positive compiler warnings disappear by setting values. */
vec3 worldPosition = vec3(0);
vec3 viewPosition = vec3(0);
vec3 worldNormal = vec3(0);
vec3 viewNormal = vec3(0);

#  elif defined(GPU_GEOMETRY_SHADER)
in ShaderStageInterface{SURFACE_INTERFACE} dataIn[];

out ShaderStageInterface{SURFACE_INTERFACE} dataOut;

#    define PASS_SURFACE_INTERFACE(vert) \
      dataOut.worldPosition = dataIn[vert].worldPosition; \
      dataOut.viewPosition = dataIn[vert].viewPosition; \
      dataOut.worldNormal = dataIn[vert].worldNormal; \
      dataOut.viewNormal = dataIn[vert].viewNormal;

#  else /* GPU_VERTEX_SHADER || GPU_FRAGMENT_SHADER*/

IN_OUT ShaderStageInterface{SURFACE_INTERFACE};

#  endif
#endif /* EEVEE_GENERATED_INTERFACE */

#ifdef HAIR_SHADER
IN_OUT ShaderHairInterface
{
  /* world space */
  vec3 hairTangent;
  float hairThickTime;
  float hairThickness;
  float hairTime;
  flat int hairStrandID;
  vec2 hairBary;
};
#endif

#ifdef POINTCLOUD_SHADER
IN_OUT ShaderPointCloudInterface
{
  /* world space */
  float pointRadius;
  float pointPosition;
  flat int pointID;
};
#endif

#if defined(GPU_FRAGMENT_SHADER) && defined(CODEGEN_LIB)

#  if defined(USE_BARYCENTRICS) && !defined(HAIR_SHADER)
vec3 barycentric_distances_get()
{
  /* NOTE: No need to undo perspective divide since it is not applied yet. */
  vec3 pos0 = (ProjectionMatrixInverse * gpu_position_at_vertex(0)).xyz;
  vec3 pos1 = (ProjectionMatrixInverse * gpu_position_at_vertex(1)).xyz;
  vec3 pos2 = (ProjectionMatrixInverse * gpu_position_at_vertex(2)).xyz;
  vec3 edge21 = pos2 - pos1;
  vec3 edge10 = pos1 - pos0;
  vec3 edge02 = pos0 - pos2;
  vec3 d21 = safe_normalize(edge21);
  vec3 d10 = safe_normalize(edge10);
  vec3 d02 = safe_normalize(edge02);
  vec3 dists;
  float d = dot(d21, edge02);
  dists.x = sqrt(dot(edge02, edge02) - d * d);
  d = dot(d02, edge10);
  dists.y = sqrt(dot(edge10, edge10) - d * d);
  d = dot(d10, edge21);
  dists.z = sqrt(dot(edge21, edge21) - d * d);
  return dists.xyz;
}
#  endif

GlobalData init_globals(void)
{
  GlobalData surf;
  surf.P = worldPosition;
  surf.N = safe_normalize(worldNormal);
  surf.Ng = safe_normalize(cross(dFdx(surf.P), dFdy(surf.P)));
  surf.barycentric_coords = vec2(0.0);
  surf.barycentric_dists = vec3(0.0);
#  ifdef HAIR_SHADER
  /* Shade as a cylinder. */
  float cos_theta = hairThickTime / hairThickness;
  float sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
  surf.N = normalize(worldNormal * sin_theta + hairTangent * cos_theta);
  surf.T = hairTangent;
  surf.is_strand = true;
  surf.hair_time = hairTime;
  surf.hair_thickness = hairThickness;
  surf.hair_strand_id = hairStrandID;
#    ifdef USE_BARYCENTRICS
  surf.barycentric_coords = hair_resolve_barycentric(hairBary);
#    endif
#  else
  surf.T = vec3(0.0);
  surf.is_strand = false;
  surf.hair_time = 0.0;
  surf.hair_thickness = 0.0;
  surf.hair_strand_id = 0;
#    ifdef USE_BARYCENTRICS
  surf.barycentric_coords = gpu_BaryCoord.xy;
  surf.barycentric_dists = barycentric_distances_get();
#    endif
#  endif
  surf.ray_type = rayType;
  surf.ray_depth = 0.0;
  surf.ray_length = distance(surf.P, cameraPos);

#  if defined(WORLD_BACKGROUND) || defined(PROBE_CAPTURE)
  surf.N = surf.Ng = surf.P = -cameraVec(g_data.P);
  surf.ray_length = 0.0;
#  endif
  return surf;
}
#endif
