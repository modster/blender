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

#if defined(STEP_RESOLVE) || defined(STEP_RAYTRACE)
/* SSR will set these global variables itself.
 * Also make false positive compiler warnings disappear by setting values. */
vec3 worldPosition = vec3(0);
vec3 viewPosition = vec3(0);
vec3 worldNormal = vec3(0);
vec3 viewNormal = vec3(0);

#elif defined(GPU_GEOMETRY_SHADER)
in ShaderStageInterface{SURFACE_INTERFACE} dataIn[];

out ShaderStageInterface{SURFACE_INTERFACE} dataOut;

#  define PASS_SURFACE_INTERFACE(vert) \
    dataOut.worldPosition = dataIn[vert].worldPosition; \
    dataOut.viewPosition = dataIn[vert].viewPosition; \
    dataOut.worldNormal = dataIn[vert].worldNormal; \
    dataOut.viewNormal = dataIn[vert].viewNormal;

#else /* GPU_VERTEX_SHADER || GPU_FRAGMENT_SHADER*/

IN_OUT ShaderStageInterface{SURFACE_INTERFACE};

#endif

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
GlobalData init_globals(void)
{
  GlobalData surf;
  surf.P = worldPosition;
  surf.N = normalize(worldNormal);
  surf.Ng = safe_normalize(cross(dFdx(surf.P), dFdy(surf.P)));
#  ifdef HAIR_SHADER
  /* Shade as a cylinder. */
  float cos_theta = hairThickTime / hairThickness;
  float sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
  surf.N = normalize(worldNormal * sin_theta + hairTangent * cos_theta);
  surf.is_strand = true;
  surf.hair_time = hairTime;
  surf.hair_thickness = hairThickness;
  surf.hair_strand_id = hairStrandID;
  surf.barycentric_coords = hair_resolve_barycentric(hairBary);
#  else
  surf.is_strand = false;
  surf.hair_time = 0.0;
  surf.hair_thickness = 0.0;
  surf.hair_strand_id = 0;
  surf.barycentric_coords = vec2(0.0); /* TODO(fclem) */
#  endif
  // surf.barycentric_dists = interp.barycentric_dists; /* TODO(fclem) */
  surf.ray_type = rayType;
  surf.ray_depth = 0.0;
  surf.ray_length = distance(surf.P, cameraPos);
  surf.closure_rand = 0.5;
  return surf;
}
#endif
