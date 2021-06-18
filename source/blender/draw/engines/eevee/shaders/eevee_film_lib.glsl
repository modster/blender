
/**
 * Film accumulation utils functions.
 **/

#pragma BLENDER_REQUIRE(eevee_shader_shared.hh)
#pragma BLENDER_REQUIRE(eevee_camera_lib.glsl)

bool film_is_color_data(FilmData film)
{
  return film.data_type < FILM_DATA_FLOAT;
}

vec4 film_data_encode(FilmData film, vec4 data, float weight)
{
  if (film_is_color_data(film)) {
    /* Could we assume safe color from earlier pass? */
    data = safe_color(data);
    /* Convert transmittance to opacity. */
    data.a = saturate(1.0 - data.a);
  }

  if (film.data_type == FILM_DATA_COLOR_LOG) {
    /* TODO(fclem) Pre-expose. */
    data.rgb = log2(1.0 + data.rgb);
  }
  else if (film.data_type == FILM_DATA_DEPTH) {
    /* TODO(fclem) Depth should be converted to radial depth in panoramic projection. */
  }
  else if (film.data_type == FILM_DATA_MOTION) {
    /* Motion vectors are in camera uv space. But final motion vectors are in pixel units. */
    data *= film.uv_scale_inv.xyxy;
  }

  if (film_is_color_data(film)) {
    data *= weight;
  }
  return data;
}

vec4 film_data_decode(FilmData film, vec4 data, float weight)
{
  if (film_is_color_data(film)) {
    data *= safe_rcp(weight);
  }

  if (film.data_type == FILM_DATA_COLOR_LOG) {
    /* TODO(fclem) undo Pre-expose. */
    data.rgb = exp2(data.rgb) - 1.0;
  }
  return data;
}

/* Returns uv's position in the previous frame. */
vec2 film_uv_history_get(CameraData camera, CameraData camera_history, vec2 uv)
{
#if 0 /* TODO reproject history */
  vec3 V = camera_view_from_uv(camera, uv);
  vec3 V_prev = transform_point(hitory_mat, V);
  vec2 uv_history = camera_uv_from_view(camera_history, V_prev);
  return uv_history;
#endif
  return uv;
}

/* -------------------------------------------------------------------- */
/** \name Filter
 * \{ */

float film_filter_weight(CameraData camera, vec2 offset)
{
#if 1 /* Faster */
  /* Gaussian fitted to Blackman-Harris. */
  float r = len_squared(offset) / sqr(camera.filter_size);
  const float sigma = 0.284;
  const float fac = -0.5 / (sigma * sigma);
  float weight = exp(fac * r);
#else
  /* Blackman-Harris filter. */
  float r = M_2PI * saturate(0.5 + length(offset) / (2.0 * camera.filter_size));
  float weight = 0.35875 - 0.48829 * cos(r) + 0.14128 * cos(2.0 * r) - 0.01168 * cos(3.0 * r);
#endif
  /* Always return a weight above 0 to avoid blind spots between samples. */
  return max(weight, 1e-6);
}

/* Camera UV is the full-frame UV. Film uv is after cropping from render border. */
vec2 film_sample_from_camera_uv(FilmData film, vec2 sample_uv)
{
  return (sample_uv - film.uv_bias) * film.uv_scale_inv;
}

vec2 film_sample_to_camera_uv(FilmData film, vec2 sample_co)
{
  return sample_co * film.uv_scale + film.uv_bias;
}

void film_process_sample(CameraData camera,
                         FilmData film,
                         mat4 input_persmat,
                         mat4 input_persinv,
                         sampler2D input_tx,
                         vec2 sample_offset,
                         inout vec4 data,
                         inout float weight)
{
  /* Project sample from destrination space to source texture. */
  vec2 sample_center = gl_FragCoord.xy;
  vec2 sample_uv = film_sample_to_camera_uv(film, sample_center + sample_offset);
  vec3 vV_dst = camera_view_from_uv(camera, sample_uv);
  /* Pixels outside of projection range. */
  if (vV_dst == vec3(0.0)) {
    return;
  }

  bool is_persp = camera.type != CAMERA_ORTHO;
  vec2 uv_src = camera_uv_from_view(input_persmat, is_persp, vV_dst);
  /* Snap to sample actual location (pixel center). */
  vec2 input_size = vec2(textureSize(input_tx, 0));
  vec2 texel_center_src = floor(uv_src * input_size) + 0.5;
  uv_src = texel_center_src / input_size;
  /* Discard pixels outside of input range. */
  if (any(greaterThan(abs(uv_src - 0.5), vec2(0.5)))) {
    return;
  }

  /* Reproject sample location in destination space to have correct distance metric. */
  vec3 vV_src = camera_view_from_uv(input_persinv, uv_src);
  vec2 uv_cam = camera_uv_from_view(camera, vV_src);
  vec2 sample_dst = film_sample_from_camera_uv(film, uv_cam);

  /* Equirectangular projection might wrap and have more than one point mapping to the same
   * original coordinate. We need to get the closest pixel center.
   * NOTE: This is wrong for projection outside the main frame. */
  if (camera.type == CAMERA_PANO_EQUIRECT) {
    sample_center = film_sample_to_camera_uv(film, sample_center);
    vec3 vV_center = camera_view_from_uv(camera, sample_center);
    sample_center = camera_uv_from_view(camera, vV_center);
    sample_center = film_sample_from_camera_uv(film, sample_center);
  }
  /* Compute filter weight and add to weighted sum. */
  vec2 offset = sample_dst - sample_center;
  float sample_weight = film_filter_weight(camera, offset);
  vec4 sample_data = textureLod(input_tx, uv_src, 0.0);
  data += film_data_encode(film, sample_data, sample_weight);
  weight += sample_weight;
}

/** \} */
