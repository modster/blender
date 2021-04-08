
/**
 * Recombine Pass: Load separate convolution layer and composite with self
 * slight defocus convolution and in-focus fields.
 *
 * The halfres gather methods are fast but lack precision for small CoC areas.
 * To fix this we do a bruteforce gather to have a smooth transition between
 * in-focus and defocus regions.
 */

#pragma BLENDER_REQUIRE(eevee_depth_of_field_accumulator_lib.glsl)

layout(std140) uniform dof_block
{
  DepthOfFieldData dof;
};

uniform sampler2D depth_tx;
uniform sampler2D color_tx;
uniform sampler2D color_bg_tx;
uniform sampler2D color_fg_tx;
uniform sampler2D color_holefill_tx;
uniform sampler2D tiles_bg_tx;
uniform sampler2D tiles_fg_tx;
uniform sampler2D weight_bg_tx;
uniform sampler2D weight_fg_tx;
uniform sampler2D weight_holefill_tx;
uniform sampler2D bokeh_lut_tx;

in vec4 uvcoordsvar;

layout(location = 0) out vec4 out_color;

void main(void)
{
  ivec2 tile_co = ivec2(gl_FragCoord.xy / float(DOF_TILE_DIVISOR));
  CocTile coc_tile = dof_coc_tile_load(tiles_fg_tx, tiles_bg_tx, tile_co);
  CocTilePrediction prediction = dof_coc_tile_prediction_get(coc_tile);

  out_color = vec4(0.0);
  float weight = 0.0;

  vec4 layer_color;
  float layer_weight;

  vec2 uv_halfres = gl_FragCoord.xy / (2.0 * vec2(textureSize(color_bg_tx, 0)));

  if (!no_holefill_pass && prediction.do_holefill) {
    layer_color = textureLod(color_holefill_tx, uv_halfres, 0.0);
    layer_weight = textureLod(weight_holefill_tx, uv_halfres, 0.0).r;
    out_color = layer_color * safe_rcp(layer_weight);
    weight = float(layer_weight > 0.0);
  }

  if (!no_background_pass && prediction.do_background) {
    layer_color = textureLod(color_bg_tx, uv_halfres, 0.0);
    layer_weight = textureLod(weight_bg_tx, uv_halfres, 0.0).r;
    /* Always prefer background to holefill pass. */
    layer_color *= safe_rcp(layer_weight);
    layer_weight = float(layer_weight > 0.0);
    /* Composite background. */
    out_color = out_color * (1.0 - layer_weight) + layer_color;
    weight = weight * (1.0 - layer_weight) + layer_weight;
    /* Fill holes with the composited background. */
    out_color *= safe_rcp(weight);
    weight = float(weight > 0.0);
  }

  if (!no_slight_focus_pass && prediction.do_slight_focus) {
    dof_slight_focus_gather(dof,
                            depth_tx,
                            color_tx,
                            bokeh_lut_tx,
                            coc_tile.fg_slight_focus_max_coc,
                            layer_color,
                            layer_weight);
    /* Composite slight defocus. */
    out_color = out_color * (1.0 - layer_weight) + layer_color;
    weight = weight * (1.0 - layer_weight) + layer_weight;
  }

  if (!no_focus_pass && prediction.do_focus) {
    layer_color = safe_color(textureLod(color_tx, uvcoordsvar.xy, 0.0));
    layer_weight = 1.0;
    /* Composite in focus. */
    out_color = out_color * (1.0 - layer_weight) + layer_color;
    weight = weight * (1.0 - layer_weight) + layer_weight;
  }

  if (!no_foreground_pass && prediction.do_foreground) {
    layer_color = textureLod(color_fg_tx, uv_halfres, 0.0);
    layer_weight = textureLod(weight_fg_tx, uv_halfres, 0.0).r;
    /* Composite foreground. */
    out_color = out_color * (1.0 - layer_weight) + layer_color;
  }

  /* Fix float precision issue in alpha compositing.  */
  if (out_color.a > 0.99) {
    out_color.a = 1.0;
  }

  if (debug_resolve_perf && coc_tile.fg_slight_focus_max_coc >= 0.5) {
    out_color.rgb *= vec3(1.0, 0.1, 0.1);
  }
}
