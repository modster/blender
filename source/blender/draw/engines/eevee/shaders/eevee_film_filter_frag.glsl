
/**
 * Accumulate input texture into the film accumulation buffer.
 *
 * All samples inside the filter radius are projected to the input texture.
 * The nearest input sample is then projected back to the destination texture space
 * to get an accurate filter weight.
 *
 * If using nearest filtering (for non-color data) only the closest sample is considered
 * and the weight is use as a distance metric.
 **/

#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_film_lib.glsl)

layout(std140) uniform camera_block
{
  CameraData camera;
};

layout(std140) uniform film_block
{
  FilmData film;
};

uniform sampler2D input_tx;
uniform sampler2D data_tx;
uniform sampler2D weight_tx;

in vec4 uvcoordsvar;

layout(location = 0) out vec4 out_data;
layout(location = 1) out float out_weight;

/* clang-format off */
const vec2 sample_offsets_plus[5] = vec2[5](vec2(0, 1), vec2(-1, 0), vec2(0, 0), vec2(1, 0), vec2(0, -1));
const vec2 sample_offsets_3x3[9] = vec2[9](vec2(-1, 1), vec2(0, 1), vec2(1, 1), vec2(-1, 0), vec2(0, 0), vec2(1, 0), vec2(-1, -1), vec2(0, -1), vec2(1, -1));
/* clang-format on */

void main(void)
{
  out_data = vec4(0.0);
  out_weight = 0.0;

  /* TODO(fclem) Split into multiple shaders? Measure benefits. */
  if (camera.filter_size < 1.0 || film.data_type != FILM_DATA_COLOR) {
    film_process_sample(camera,
                        film,
                        ViewProjectionMatrix,
                        ViewProjectionMatrixInverse,
                        input_tx,
                        vec2(0.0),
                        out_data,
                        out_weight);
  }
  else if (camera.filter_size < M_SQRT2) {
    for (int i = 0; i < 5; i++) {
      film_process_sample(camera,
                          film,
                          ViewProjectionMatrix,
                          ViewProjectionMatrixInverse,
                          input_tx,
                          sample_offsets_plus[i],
                          out_data,
                          out_weight);
    }
  }
  else if (camera.filter_size < 2.0) {
    for (int i = 0; i < 9; i++) {
      film_process_sample(camera,
                          film,
                          ViewProjectionMatrix,
                          ViewProjectionMatrixInverse,
                          input_tx,
                          sample_offsets_3x3[i],
                          out_data,
                          out_weight);
    }
  }
  else {
    /* This is slow but using large filter is not very common. */
    float extent = floor(camera.filter_size);
    for (float x = -extent; x < extent; x++) {
      for (float y = -extent; y < extent; y++) {
        film_process_sample(camera,
                            film,
                            ViewProjectionMatrix,
                            ViewProjectionMatrixInverse,
                            input_tx,
                            vec2(x, y),
                            out_data,
                            out_weight);
      }
    }
  }

  if (film.use_history) {
    vec2 uv_history = film_uv_history_get(camera, camera, uvcoordsvar.xy);
    vec4 history_data = textureLod(data_tx, uv_history, 0.0);
    float history_weight = textureLod(weight_tx, uv_history, 0.0).r;

    if (film.data_type == FILM_DATA_COLOR) {
      out_data += history_data;
      out_weight += history_weight;
    }
    else {
      /* Non-color data do not accumulates. It is replaced by nearest value. */
      if (history_weight > out_weight) {
        out_weight = history_weight;
        out_data = history_data;
      }
    }
  }
}
