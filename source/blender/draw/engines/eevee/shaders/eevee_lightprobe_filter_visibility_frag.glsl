
/**
 * Outputs convolved visibility from an input depth cubemap.
 * The output is an octahedral map which encodes depth in 4 components of 1 byte each.
 */

#pragma BLENDER_REQUIRE(common_math_geom_lib.glsl)
#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_bsdf_sampling_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_irradiance_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_lightprobe_filter_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_sampling_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shader_shared.hh)

uniform samplerCube depth_tx;

layout(std140) uniform filter_block
{
  LightProbeFilterData probe;
};

layout(location = 0) out vec4 out_irradiance;

vec3 octahedral_to_cubemap_proj(vec2 co)
{
  co = co * 2.0 - 1.0;

  vec2 abs_co = abs(co);
  vec3 v = vec3(co, 1.0 - (abs_co.x + abs_co.y));

  if (abs_co.x + abs_co.y > 1.0) {
    v.xy = (abs(co.yx) - 1.0) * -sign(co.xy);
  }

  return v;
}

float get_world_distance(float depth, vec3 coords)
{
  float is_background = step(1.0, depth);
  depth = get_view_z_from_depth(depth);
  depth += 1e1 * is_background;
  coords = normalize(abs(coords));
  float cos_vec = max_v3(coords);
  return depth / cos_vec;
}

void main()
{
  vec2 uv = interp.coord.xy;
  vec2 stored_texel_size = dFdx(uv);
  /* Add a 1 pixel border all around the octahedral map to ensure filtering is correct. */
  uv = (uv - stored_texel_size) / (1.0 - 2.0 * stored_texel_size);
  /* Edge mirroring : only mirror if directly adjacent (not diagonally adjacent). */
  vec2 m = abs(uv - 0.5) + 0.5;
  vec2 f = floor(m);
  if (f.x - f.y != 0.0) {
    uv = 1.0 - uv;
  }
  uv = fract(uv);

  vec3 T, B, N;
  N = normalize(octahedral_to_cubemap_proj(uv));
  make_orthonormal_basis(N, T, B);

  vec2 accum = vec2(0.0);

  for (float i = 0; i < probe.sample_count; i++) {
    vec3 Xi = sample_cylinder(hammersley_2d(i, probe.sample_count));

    vec3 dir = sample_uniform_cone(Xi, M_PI_2 * probe.visibility_blur, N, T, B);
    float depth = texture(depth_tx, dir).r;
    depth = get_world_distance(depth, dir);
    accum += vec2(depth, depth * depth);
  }

  out_irradiance = visibility_encode(abs(accum / probe.sample_count), probe.visibility_range);
}
