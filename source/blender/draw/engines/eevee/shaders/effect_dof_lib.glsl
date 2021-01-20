
#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(common_math_lib.glsl)

uniform vec4 cocParams;

#define cocMul cocParams[0] /* distance * aperturesize * invsensorsize */
#define cocBias cocParams[1] /* aperturesize * invsensorsize */
#define cocNear cocParams[2] /* Near view depths value. */
#define cocFar cocParams[3] /* Far view depths value. */

uniform float scatterColorThreshold;
uniform float scatterCocThreshold;

/* -------------- Utils ------------- */

/* Allow 5% CoC error. */
#define DOF_FAST_GATHER_COC_ERROR 0.05

const vec2 quad_offsets[4] = vec2[4](
    vec2(-0.5, 0.5), vec2(0.5, 0.5), vec2(0.5, -0.5), vec2(-0.5, -0.5));

/* Divide by sensor size to get the normalized size. */
#define calculate_coc(zdepth) (cocMul / zdepth - cocBias)

/* Ortho conversion is only true for camera view! */
#define linear_depth_persp(d) ((cocNear * cocFar) / (d * (cocNear - cocFar) + cocFar))
#define linear_depth_ortho(d) (d * (cocFar - cocNear) + cocNear)

#define linear_depth(d) \
  ((ProjectionMatrix[3][3] == 0.0) ? linear_depth_persp(d) : linear_depth_ortho(d))

#define dof_coc_from_zdepth(d) calculate_coc(linear_depth(d))

vec4 safe_color(vec4 c)
{
  /* Clamp to avoid black square artifacts if a pixel goes NaN. */
  return clamp(c, vec4(0.0), vec4(1e20)); /* 1e20 arbitrary. */
}

float dof_hdr_color_weight(vec4 color)
{
  /* From UE4. Very fast "luma" weighting. */
  float luma = (color.g * 2.0) + (color.r + color.b);
  /* TODO(fclem) Pass correct exposure. */
  const float exposure = 1.0;
  return 1.0 / (luma * exposure + 4.0);
}

float dof_coc_select(vec4 cocs)
{
  /* Select biggest coc. */
  float selected_coc = cocs.x;
  if (abs(cocs.y) > abs(selected_coc)) {
    selected_coc = cocs.y;
  }
  if (abs(cocs.z) > abs(selected_coc)) {
    selected_coc = cocs.z;
  }
  if (abs(cocs.w) > abs(selected_coc)) {
    selected_coc = cocs.w;
  }
  return selected_coc;
}

/* NOTE: Do not forget to normalize weights afterwards. */
vec4 dof_downsample_bilateral_coc_weights(vec4 cocs)
{
  float chosen_coc = dof_coc_select(cocs);

  const float scale = 4.0; /* TODO(fclem) revisit. */
  /* NOTE: The difference between the cocs should be inside a abs() function,
   * but we follow UE4 implementation to improve how dithered transparency looks (see slide 19). */
  return saturate(1.0 - (chosen_coc - cocs) * scale);
}

/* NOTE: Do not forget to normalize weights afterwards. */
vec4 dof_downsample_bilateral_color_weights(vec4 colors[4])
{
  vec4 weights;
  for (int i = 0; i < 4; i++) {
    weights[i] = dof_hdr_color_weight(colors[i]);
  }
  return weights;
}

/* Makes sure the load functions distribute the energy correctly
 * to both scatter and gather passes. */
vec4 dof_load_gather_color(sampler2D gather_input_color_buffer, vec2 uv, float lod)
{
  vec4 color = textureLod(gather_input_color_buffer, uv, lod);
  color.rgb = abs(max(color.rgb, -scatterColorThreshold));
  return color;
}

vec4 dof_load_scatter_color(sampler2D gather_input_color_buffer, vec2 uv, float lod)
{
  vec4 color = textureLod(gather_input_color_buffer, uv, lod);
  color.rgb = max(-color.rgb - scatterColorThreshold, 0.0);
  return color;
}

float dof_load_gather_coc(sampler2D gather_input_coc_buffer, vec2 uv, float lod)
{
  float coc = textureLod(gather_input_coc_buffer, uv, lod).r;
  /* We gather at halfres. CoC must be divided by 2 to be compared against radii. */
  return coc * 0.5;
}

/* Distribute weights between near/slightfocus/far fields (slide 117). */
const float layer_threshold = 4.0;
/* For some reason 0.5 is not enough to make it watertight. */
const float layer_offset = 0.5 + 0.2;

float dof_layer_weight(float coc, const bool is_foreground)
{
/* NOTE: These are fullres pixel CoC value. */
#ifdef DOF_RESOLVE_PASS
  return saturate(-abs(coc) + layer_threshold + layer_offset) *
         float(is_foreground ? (coc <= 0.5) : (coc > -0.5));
#else
  coc *= 2.0; /* Account for half pixel gather. */
  return saturate(((is_foreground) ? -coc : coc) - layer_threshold + layer_offset);
#endif
}
vec4 dof_layer_weight(vec4 coc)
{
  /* NOTE: Used for scatter pass which already flipped the sign correctly. */
  coc *= 2.0; /* Account for half pixel gather. */
  return saturate(coc - layer_threshold + layer_offset);
}

/* NOTE: This is halfres CoC radius. */
float dof_sample_weight(float coc)
{
  /* Full intensity if CoC radius is below the pixel footprint. */
  const float min_coc = 1.0;
  coc = max(min_coc, coc);
  return (M_PI * min_coc * min_coc) / (M_PI * coc * coc);
}
vec4 dof_sample_weight(vec4 coc)
{
  /* Full intensity if CoC radius is below the pixel footprint. */
  const float min_coc = 1.0;
  coc = max(vec4(min_coc), coc);
  return (M_PI * min_coc * min_coc) / (M_PI * coc * coc);
}

/* Intersection with the center of the kernel. */
float dof_intersection_weight(float coc, float distance_from_center)
{
  return saturate((abs(coc) - distance_from_center) * 1.0 + 0.5);
}

/* Returns weight of the sample for the outer bucket (containing previous rings). */
float dof_gather_accum_weight(float coc, float bordering_radius, bool first_ring)
{
  /* First ring has nothing to be mixed against. */
  if (first_ring) {
    return 0.0;
  }
  return saturate(coc - bordering_radius);
}

/* ------------------- COC TILES UTILS ------------------- */

struct CocTile {
  float fg_min_coc;
  float fg_max_coc;
  float fg_slight_focus_max_coc;
  float bg_min_coc;
  float bg_max_coc;
  float bg_min_intersectable_coc;
};

/* WATCH: Might have to change depending on the texture format. */
#define DOF_TILE_DEFOCUS 0.25
#define DOF_TILE_FOCUS 0.0
#define DOF_TILE_LARGE_COC 1024.0

/* Init a CoC tile for reduction algorithms. */
CocTile dof_coc_tile_init(void)
{
  CocTile tile;
  tile.fg_min_coc = 0.0;
  tile.fg_max_coc = -DOF_TILE_LARGE_COC;
  tile.fg_slight_focus_max_coc = -1.0;
  tile.bg_min_coc = DOF_TILE_LARGE_COC;
  tile.bg_max_coc = 0.0;
  tile.bg_min_intersectable_coc = DOF_TILE_LARGE_COC;
  return tile;
}

CocTile dof_coc_tile_load(sampler2D fg_buffer, sampler2D bg_buffer, ivec2 tile_co)
{
  ivec2 tex_size = textureSize(fg_buffer, 0).xy;
  tile_co = clamp(tile_co, ivec2(0), tex_size - 1);

  vec3 fg = texelFetch(fg_buffer, tile_co, 0).xyz;
  vec3 bg = texelFetch(bg_buffer, tile_co, 0).xyz;

  CocTile tile;
  tile.fg_min_coc = -fg.x;
  tile.fg_max_coc = -fg.y;
  tile.fg_slight_focus_max_coc = fg.z;
  tile.bg_min_coc = bg.x;
  tile.bg_max_coc = bg.y;
  tile.bg_min_intersectable_coc = bg.z;
  return tile;
}

/* ------------------- GATHER UTILS ------------------- */

struct DofGatherData {
  vec4 color;
  float weight;
  float dist; /* TODO remove */
  /* For scatter occlusion. */
  float coc;
  float coc_sqr;
  /* For ring bucket merging. */
  float transparency;

  float layer_opacity;
};

#define GATHER_DATA_INIT DofGatherData(vec4(0.0), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

void dof_gather_accumulate_sample(DofGatherData sample_data,
                                  float weight,
                                  inout DofGatherData accum_data)
{
  accum_data.color += sample_data.color * weight;
  accum_data.coc += sample_data.coc * weight;
  accum_data.coc_sqr += sample_data.coc * (sample_data.coc * weight);
  accum_data.weight += weight;
}

void dof_gather_accumulate_sample_pair(DofGatherData pair_data[2],
                                       float bordering_radius,
                                       bool first_ring,
                                       const bool do_fast_gather,
                                       const bool is_foreground,
                                       inout DofGatherData ring_data,
                                       inout DofGatherData accum_data)
{
  if (do_fast_gather) {
    for (int i = 0; i < 2; i++) {
      dof_gather_accumulate_sample(pair_data[i], 1.0, accum_data);
      accum_data.layer_opacity += 1.0;
    }
    return;
  }

  float inter_weight[2];

  for (int i = 0; i < 2; i++) {
    inter_weight[i] = dof_intersection_weight(pair_data[i].coc, pair_data[i].dist);
  }

#ifdef DOF_HOLEFILL_PASS
  if (pair_data[0].coc < 0.0 && pair_data[1].coc > pair_data[0].coc) {
    inter_weight[1] = max(inter_weight[1], inter_weight[0]);
  }
  else if (pair_data[1].coc < 0.0 && pair_data[0].coc > pair_data[1].coc) {
    inter_weight[0] = max(inter_weight[1], inter_weight[0]);
  }
#endif

  for (int i = 0; i < 2; i++) {
#ifdef DOF_HOLEFILL_PASS
    float layer_weight = float(pair_data[i].coc > -layer_threshold);
#else
    float layer_weight = dof_layer_weight(pair_data[i].coc, is_foreground);
#endif
    float sample_weight = dof_sample_weight(pair_data[i].coc);
    float weight = inter_weight[i] * layer_weight * sample_weight;

    /**
     * If a CoC is larger than bordering radius we accumulate it to the general accumulator.
     * If not, we accumulate to the ring bucket. This is to have more consistent sample occlusion.
     **/
    float accum_weight = dof_gather_accum_weight(pair_data[i].coc, bordering_radius, first_ring);
    dof_gather_accumulate_sample(pair_data[i], weight * accum_weight, accum_data);
    dof_gather_accumulate_sample(pair_data[i], weight * (1.0 - accum_weight), ring_data);

    accum_data.layer_opacity += layer_weight;

    float coc = is_foreground ? -pair_data[i].coc : pair_data[i].coc;
    ring_data.transparency += saturate(coc - bordering_radius);
  }
}

void dof_gather_accumulate_sample_ring(DofGatherData ring_data,
                                       int sample_count,
                                       bool first_ring,
                                       const bool do_fast_gather,
                                       /* accum_data occludes the ring_data if true. */
                                       const bool reversed_occlusion,
                                       inout DofGatherData accum_data)
{
  if (do_fast_gather) {
    /* Do nothing as ring_data contains nothing. All samples are already in accum_data. */
    return;
  }

  if (first_ring) {
    /* Layer opacity is directly accumulated into accum_data data. */
    accum_data.color = ring_data.color;
    accum_data.coc = ring_data.coc;
    accum_data.coc_sqr = ring_data.coc_sqr;
    accum_data.weight = ring_data.weight;

    accum_data.transparency = ring_data.transparency;
    return;
  }

  if (ring_data.weight == 0.0) {
    return;
  }

  float ring_avg_coc = ring_data.coc / ring_data.weight;
  float accum_avg_coc = accum_data.coc / accum_data.weight;

  /* Smooth test to set opacity to see if the ring average coc occludes the accumulation.
   * Test is reversed to be multiplied against opacity. */
  float ring_no_occlu = saturate(accum_avg_coc - ring_avg_coc);
  float accum_no_occlu = saturate(ring_avg_coc - accum_avg_coc + 1.0);

  /* (Slide 40) */
  float ring_opacity = saturate(1.0 - ring_data.transparency / float(sample_count));
  float accum_opacity = 1.0 - accum_data.transparency;

  if (reversed_occlusion) {
    /* Accum_data occludes the ring. */
    float alpha = (accum_data.weight == 0.0) ? 0.0 : (1.0 - accum_opacity * accum_no_occlu);
    alpha = 1.0 - alpha;

    accum_data.color += ring_data.color * alpha;
    accum_data.coc += ring_data.coc * alpha;
    accum_data.coc_sqr += ring_data.coc_sqr * alpha;
    accum_data.weight += ring_data.weight * alpha;

    accum_data.transparency += (1.0 - ring_opacity) * alpha;
  }
  else {
    /* Ring occludes the accum_data (Same as reference). */
    float alpha = (accum_data.weight == 0.0) ? 0.0 : (1.0 - ring_opacity * ring_no_occlu);

    accum_data.color = accum_data.color * alpha + ring_data.color;
    accum_data.coc = accum_data.coc * alpha + ring_data.coc;
    accum_data.coc_sqr = accum_data.coc_sqr * alpha + ring_data.coc_sqr;
    accum_data.weight = accum_data.weight * alpha + ring_data.weight;
  }
}

void dof_gather_accumulate_center_sample(DofGatherData center_data,
                                         float bordering_radius,
                                         const bool do_fast_gather,
                                         const bool is_foreground,
                                         inout DofGatherData accum_data)
{
  float layer_weight = dof_layer_weight(center_data.coc, is_foreground);
  float sample_weight = dof_sample_weight(center_data.coc);
  float weight = layer_weight * sample_weight;

  if (do_fast_gather) {
    /* Hope for the compiler to optimize the above. */
    layer_weight = 1.0;
    sample_weight = 1.0;
    weight = 1.0;
  }

  float coc = is_foreground ? -center_data.coc : center_data.coc;
  center_data.transparency = saturate(coc - bordering_radius);

  /* Fix an issue with last ring opacity not contributing enough. */
  if (is_foreground) {
    center_data.transparency = 0.0;
  }

  /* Logic of dof_gather_accumulate_sample */
  center_data.coc_sqr = center_data.coc * (center_data.coc * weight);
  center_data.color *= weight;
  center_data.coc *= weight;
  center_data.weight = weight;

  accum_data.layer_opacity += layer_weight;

  /* Accumulate center as its own rign. */
  dof_gather_accumulate_sample_ring(
      center_data, 1, false, do_fast_gather, is_foreground, accum_data);
}

float dof_gather_total_sample_count(const int ring_count)
{
  float float_ring = float(ring_count);
  return ((float_ring * float_ring - float_ring) * 2.0 + 1.0) * 2.0;
}

void dof_gather_accumulate_resolve(const int ring_count,
                                   DofGatherData accum_data,
                                   out vec4 out_col,
                                   out float out_weight,
                                   out vec2 out_occlusion)
{
  float weight_inv = safe_rcp(accum_data.weight);
  out_col = accum_data.color * weight_inv;
  out_occlusion = vec2(abs(accum_data.coc), accum_data.coc_sqr) * weight_inv;

  if (accum_data.weight > 0.0) {
    out_weight = accum_data.layer_opacity * safe_rcp(dof_gather_total_sample_count(ring_count));
    /* Gathering may not accumulate to 1.0 alpha because of float precision. */
    if (out_weight > 0.99) {
      out_weight = 1.0;
    }
  }
  else {
    /* Avoid dark fringe since the gather area can be larger than actual radius. */
    out_weight = 0.0;
  }
}

ivec2 dof_square_ring_sample_offset(int ring_distance, int sample_id)
{
  /**
   * Generate samples in a square pattern with the ring radius. X is the center tile.
   *
   *    Dist1          Dist2
   *                 6 5 4 3 2
   *    3 2 1        7       1
   *    . X 0        .   X   0
   *    . . .        .       .
   *                 . . . . .
   *
   * Samples are expected to be mirrored to complete the pattern.
   **/
  ivec2 offset;
  if (sample_id < ring_distance) {
    offset.x = ring_distance;
    offset.y = sample_id;
  }
  else if (sample_id < ring_distance * 3) {
    offset.x = ring_distance - sample_id + ring_distance;
    offset.y = ring_distance;
  }
  else {
    offset.x = -ring_distance;
    offset.y = ring_distance - sample_id + 3 * ring_distance;
  }
  return offset;
}