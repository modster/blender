
/**
 * Holefill pass: Gather background parts where foreground is present.
 *
 * Using the min&max CoC tile buffer, we select the best apropriate method to blur the scene color.
 * A fast gather path is taken if there is not many CoC variation inside the tile.
 *
 * We sample using an octaweb sampling pattern. We randomize the kernel center and each ring
 * rotation to ensure maximum coverage.
 **/

#pragma BLENDER_REQUIRE(eevee_depth_of_field_accumulator_lib.glsl)

layout(std140) uniform sampling_block
{
  SamplingData sampling;
};

layout(std140) uniform dof_block
{
  DepthOfFieldData dof;
};

uniform sampler2D color_tx;
uniform sampler2D color_bilinear_tx;
uniform sampler2D coc_tx;
uniform sampler2D tiles_fg_tx;
uniform sampler2D tiles_bg_tx;

layout(location = 0) out vec4 out_color;
layout(location = 1) out float out_weight;

void main()
{
  ivec2 tile_co = ivec2(gl_FragCoord.xy / float(DOF_TILE_DIVISOR / 2));
  CocTile coc_tile = dof_coc_tile_load(tiles_fg_tx, tiles_bg_tx, tile_co);
  CocTilePrediction prediction = dof_coc_tile_prediction_get(coc_tile);

  float base_radius = -coc_tile.fg_min_coc;
  float min_radius = -coc_tile.fg_max_coc;
  float min_intersectable_radius = dof_tile_large_coc;
  bool can_early_out = !prediction.do_holefill;

  bool do_fast_gather = dof_do_fast_gather(base_radius, min_radius, is_foreground);

  /* Gather at half resolution. Divide CoC by 2. */
  base_radius *= 0.5;
  min_intersectable_radius *= 0.5;

  bool do_density_change = dof_do_density_change(base_radius, min_intersectable_radius);

  if (can_early_out) {
    /* Early out. */
    out_color = vec4(0.0);
    out_weight = 0.0;
  }
  else if (do_fast_gather) {
    vec2 unused_occlusion;
    dof_gather_accumulator(sampling,
                           dof,
                           color_tx,
                           color_bilinear_tx,
                           coc_tx,
                           coc_tx,
                           base_radius,
                           min_intersectable_radius,
                           true,
                           false,
                           out_color,
                           out_weight,
                           unused_occlusion);
  }
  else {
    vec2 unused_occlusion;
    dof_gather_accumulator(sampling,
                           dof,
                           color_tx,
                           color_bilinear_tx,
                           coc_tx,
                           coc_tx,
                           base_radius,
                           min_intersectable_radius,
                           false,
                           false,
                           out_color,
                           out_weight,
                           unused_occlusion);
  }
}
