
void node_eevee_specular(vec4 diffuse,
                         vec4 specular,
                         float roughness,
                         vec4 emissive,
                         float transp,
                         vec3 normal,
                         float clearcoat,
                         float clearcoat_roughness,
                         vec3 clearcoat_normal,
                         float occlusion,
                         out Closure result)
{
  g_diffuse_data.color = diffuse.rgb;
  g_diffuse_data.N = normal;

  g_reflection_data.color = specular.rgb;
  g_reflection_data.N = normal;
  g_reflection_data.roughness = roughness;

  g_emission_data.emission = emissive.rgb;
}
