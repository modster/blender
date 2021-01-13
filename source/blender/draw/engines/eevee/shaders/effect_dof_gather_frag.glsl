
/**
 * Gather pass: Convolve foreground and background parts in separate passes.
 *
 * Using the min&max CoC tile buffer, we select the best apropriate method to blur the scene color.
 * A fast gather path is taken if there is not many CoC variation inside the tile.
 *
 * We sample using an octaweb sampling pattern. We randomize the kernel center and each ring
 * rotation to ensure maximum coverage.
 **/

#pragma BLENDER_REQUIRE(common_utiltex_lib.glsl)
#pragma BLENDER_REQUIRE(effect_dof_lib.glsl)

/* Mipmapped input buffers, halfres but with padding to ensure mipmap alignement. */
uniform sampler2D colorBuffer;
uniform sampler2D cocBuffer;

/* CoC Min&Max tile buffer at 1/16th of fullres. */
uniform sampler2D cocTilesBuffer;

/* Used to correct the padding in the color and CoC buffers. */
uniform vec2 gatherInputUvCorrection;

uniform vec2 gatherOutputTexelSize;

layout(location = 0) out vec4 outColor;
layout(location = 1) out float outWeight;

const bool is_foreground = FOREGROUND_PASS;

void dof_gather_accumulator(float base_radius, const bool do_fast_gather)
{
  vec4 noise = texelfetch_noise_tex(gl_FragCoord.xy);

#if 0 /* For debugging sample positions. */
  noise.xy = vec2(0.0, 0.0);
  noise.zw = vec2(0.0, 1.0);
#endif

  const int ring_count = 3; /* TODO(fclem) Shader variations? */
  float unit_ring_radius = 1.0 / float(ring_count);
  float unit_sample_radius = 1.0 / float(ring_count + 0.5);
  float lod = floor(log2(base_radius * unit_sample_radius) - 1.5);

  /* Jitter center half a pixel to reduce undersampling and nearest interpolation mode. */
  vec2 jitter_ofs = 0.95 * noise.zw * sqrt(noise.x);
  vec2 center_co = gl_FragCoord.xy + jitter_ofs;

  bool first_ring = true;

  DofGatherData accum_data = GATHER_DATA_INIT;

  /* TODO(fclem) another seed? For now Cranly-Partterson rotation with golden ratio. */
  noise.x = fract(noise.x + 0.61803398875);
  /* Randomize ring radius to avoid seeing the ring shapes. 1 is to not overlap the center. */
  float ring_offset = 1.0 - (noise.x) * unit_ring_radius * base_radius;

  for (int ring = ring_count; ring > 0; ring--) {
    int sample_pair_count = 4 * ring;

    float step_rot = M_PI / float(sample_pair_count);
    mat2 step_rot_mat = rot2_from_angle(step_rot);

    float ring_radius = float(ring) * unit_ring_radius * base_radius + ring_offset;

    float angle_offset = step_rot * noise.y;
    vec2 offset = vec2(cos(angle_offset), sin(angle_offset)) * ring_radius;

    /* Slide 38. */
    const float coc_radius_error = 1.0;
    float bordering_radius = ring_radius +
                             (1.5 + coc_radius_error) * base_radius * unit_sample_radius;

    DofGatherData ring_data = GATHER_DATA_INIT;
    for (int sample_pair = 0; sample_pair < sample_pair_count; sample_pair++) {
      offset = step_rot_mat * offset;

      DofGatherData pair_data[2];
      for (int i = 0; i < 2; i++) {
        vec2 sample_co = center_co + ((i == 0) ? offset : -offset);
        vec2 sample_uv = sample_co * gatherOutputTexelSize * gatherInputUvCorrection;
        pair_data[i].color = dof_load_gather_color(colorBuffer, sample_uv, lod);
        pair_data[i].coc = dof_load_gather_coc(cocBuffer, sample_uv, lod, is_foreground);
        pair_data[i].dist = distance(center_co, sample_co);
      }

      dof_gather_accumulate_sample_pair(pair_data,
                                        bordering_radius,
                                        first_ring,
                                        do_fast_gather,
                                        is_foreground,
                                        ring_data,
                                        accum_data);
    }

    dof_gather_accumulate_sample_ring(
        ring_data, sample_pair_count, first_ring, do_fast_gather, is_foreground, accum_data);

    first_ring = false;
  }

  {
    /* Center sample. */
    vec2 sample_uv = center_co * gatherOutputTexelSize * gatherInputUvCorrection;
    DofGatherData center_data;
    center_data.color = dof_load_gather_color(colorBuffer, sample_uv, lod);
    center_data.coc = dof_load_gather_coc(cocBuffer, sample_uv, lod, is_foreground);
    center_data.dist = 0.0;

    dof_gather_accumulate_center_sample(center_data, do_fast_gather, is_foreground, accum_data);
  }

  dof_gather_accumulate_resolve(ring_count, accum_data, outColor, outWeight);
}

void main()
{
  ivec2 tile_co = ivec2(gl_FragCoord.xy / 8.0);
  CocTile coc_tile = dof_coc_tile_load(cocTilesBuffer, tile_co, is_foreground);

  /* Gather at half resolution. Divide coc by 2. */
  float base_radius = 0.5 * max(0.0, is_foreground ? -coc_tile.fg_min_coc : coc_tile.bg_max_coc);
  float min_radius = 0.5 * max(0.0, is_foreground ? -coc_tile.fg_max_coc : coc_tile.bg_min_coc);

  /* Allow for a 5% radius difference. */
  bool do_fast_gather = (base_radius - min_radius) < (0.05 * base_radius);

  if (base_radius < 1.0) {
    /* Early out. */
    outColor = vec4(0.0);
    outWeight = 0.0;
  }
  else if (do_fast_gather) {
    /* Fast gather */
    dof_gather_accumulator(base_radius, true);
  }
  else {
    dof_gather_accumulator(base_radius, false);
  }
#if 0 /* Debug. */
  if (do_fast_gather) {
    outColor.rgb = outColor.rgb * vec3(0.5, 1.0, 0.5) + vec3(0.5, 1.0, 0.5) * 0.1;
  }
#endif
}