
#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_object_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_culling_iter_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shader_shared.hh)

layout(std140) uniform lights_block
{
  LightData lights[CULLING_ITEM_BATCH];
};

layout(std140) uniform lights_culling_block
{
  CullingData light_culling;
};

uniform usampler2D lights_culling_tx;

layout(location = 0, index = 0) out vec4 outRadiance;
layout(location = 0, index = 1) out vec4 outTransmittance;

MeshData g_surf;
ivec4 g_closure_data[8];

vec3 light_simple(LightData ld, vec4 l_vector)
{
  float power = 1.0;
  if (ld.type != LIGHT_SUN) {
    /**
     * Using "Point Light Attenuation Without Singularity" from Cem Yuksel
     * http://www.cemyuksel.com/research/pointlightattenuation/pointlightattenuation.pdf
     * http://www.cemyuksel.com/research/pointlightattenuation/
     **/
    float d = l_vector.w;
    float d_sqr = sqr(d);
    float r_sqr = 0.01;
    /* Using reformulation that has better numerical percision. */
    power = 2.0 / (d_sqr + r_sqr + d * sqrt(d_sqr + r_sqr));

    if (is_area_light(ld.type)) {
      /* Modulate by light plane orientation / solid angle. */
      power *= saturate(dot(ld._back, l_vector.xyz / l_vector.w));
    }
  }
  return ld.color * power;
}

void main(void)
{
  g_surf = init_from_interp();

  /* Prevent precision issues on unit coordinates. */
  vec3 p = (g_surf.P + 0.000001) * 0.999999;
  int xi = int(abs(floor(p.x)));
  int yi = int(abs(floor(p.y)));
  int zi = int(abs(floor(p.z)));
  bool check = ((mod(xi, 2) == mod(yi, 2)) == bool(mod(zi, 2)));

  float vP_z = get_view_z_from_depth(gl_FragCoord.z);

  vec3 radiance = vec3(0);
  ITEM_FOREACH_BEGIN (light_culling, lights_culling_tx, vP_z, l_idx) {
    LightData light = lights[l_idx];
    vec4 l_vector;
    l_vector.xyz = light._position - g_surf.P;
    l_vector.w = length(l_vector.xyz);
    radiance += saturate(dot(g_surf.N, l_vector.xyz / l_vector.w)) *
                light_simple(light, l_vector) * light.volume_power;
  }
  ITEM_FOREACH_END

  outRadiance = vec4(radiance * mix(0.2, 0.8, check), 1.0);
  outTransmittance = vec4(0.0, 0.0, 0.0, 1.0);
}