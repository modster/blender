
void generated_from_orco(vec3 orco, out vec3 generated)
{
#ifdef VOLUMETRICS
#  ifdef MESH_SHADER
  generated = volumeObjectLocalCoord;
#  else
  generated = g_data.P;
#  endif
#else
  generated = orco;
#endif
}

void generated_texco(vec3 attr_orco, out vec3 generated)
{
#if defined(WORLD_BACKGROUND) || defined(PROBE_CAPTURE)
  generated = normalize(g_data.P);
#else
  generated_from_orco(attr_orco, generated);
#endif
}

void node_tex_coord(mat4 obmatinv,
                    vec3 attr_orco,
                    vec3 attr_uv,
                    out vec3 generated,
                    out vec3 normal,
                    out vec3 uv,
                    out vec3 object,
                    out vec3 camera,
                    out vec3 window,
                    out vec3 reflection)
{
  vec3 N = safe_normalize(g_data.N);

  generated = attr_orco;
  normal = normal_world_to_object(N);
  uv = attr_uv;

  object = transform_point((obmatinv[3][3] == 0.0) ? ModelMatrixInverse : obmatinv, g_data.P);

  camera = transform_point(ViewMatrix, g_data.P);
  camera.z = -camera.z;
  /* TODO fix in panoramic view. */
  window.xy = project_point(ViewProjectionMatrix, g_data.P).xy * 0.5 + 0.5;
  window.xy = window.xy * CameraTexCoFactors.xy + CameraTexCoFactors.zw;
  window.z = 0.0;

  reflection = -reflect(cameraVec(g_data.P), N);
}
