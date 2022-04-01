
void node_bsdf_translucent(vec4 color, vec3 N, float weight, out Closure result)
{
  N = safe_normalize(N);

  ClosureDiffuse translucent_data;
  translucent_data.weight = weight;
  translucent_data.color = color.rgb;
  translucent_data.N = -N;
  translucent_data.sss_id = 0u;

  result = closure_eval(translucent_data);
}
