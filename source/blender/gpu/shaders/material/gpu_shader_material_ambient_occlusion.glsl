void node_ambient_occlusion(vec4 color,
                            float dist,
                            vec3 normal,
                            const float inverted,
                            const float sample_count,
                            out vec4 result_color,
                            out float result_ao)
{
  vec3 bent_normal;
  vec4 rand = texelfetch_noise_tex(gl_FragCoord.xy);
  OcclusionData data = occlusion_search(viewPosition, maxzBuffer, dist, inverted, sample_count);

  vec3 V = cameraVec(g_data.P);
  vec3 N = normalize(normal);
  vec3 Ng = safe_normalize(cross(dFdx(g_data.P), dFdy(g_data.P)));

  float unused_error;
  vec3 unused;
  occlusion_eval(data, V, N, Ng, inverted, result_ao, unused_error, unused);
  result_color = result_ao * color;
}
