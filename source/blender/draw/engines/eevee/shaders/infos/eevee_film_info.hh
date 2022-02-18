
#include "gpu_shader_create_info.hh"

/* -------------------------------------------------------------------- */
/** \name Film Filter
 * \{ */

GPU_SHADER_CREATE_INFO(eevee_film_filter)
    .do_static_compilation(true)
    .uniform_buf(2, "CameraData", "camera")
    .uniform_buf(1, "FilmData", "film")
    .sampler(0, ImageType::FLOAT_2D, "input_tx")
    .sampler(1, ImageType::FLOAT_2D, "data_tx")
    .sampler(2, ImageType::FLOAT_2D, "weight_tx")
    .fragment_out(0, Type::VEC4, "out_data")
    .fragment_out(1, Type::FLOAT, "out_weight")
    .additional_info("eevee_shared")
    .fragment_source("eevee_film_filter_frag.glsl")
    .additional_info("draw_fullscreen", "draw_view");

GPU_SHADER_CREATE_INFO(eevee_film_filter_depth)
    .do_static_compilation(true)
    .uniform_buf(2, "CameraData", "camera")
    .uniform_buf(1, "FilmData", "film")
    .sampler(0, ImageType::DEPTH_2D, "input_tx")
    .sampler(1, ImageType::FLOAT_2D, "data_tx")
    .sampler(2, ImageType::FLOAT_2D, "weight_tx")
    .fragment_out(0, Type::VEC4, "out_data")
    .fragment_out(1, Type::FLOAT, "out_weight")
    .additional_info("eevee_shared")
    .fragment_source("eevee_film_filter_frag.glsl")
    .additional_info("draw_fullscreen", "draw_view");

/** \} */

/* -------------------------------------------------------------------- */
/** \name Film Resolve
 * \{ */

GPU_SHADER_CREATE_INFO(eevee_film_resolve)
    .do_static_compilation(true)
    .uniform_buf(1, "FilmData", "film")
    .sampler(0, ImageType::FLOAT_2D, "data_tx")
    .sampler(1, ImageType::FLOAT_2D, "weight_tx")
    .sampler(2, ImageType::FLOAT_2D, "first_sample_tx")
    .fragment_out(0, Type::VEC4, "out_color")
    .additional_info("eevee_shared")
    .fragment_source("eevee_film_resolve_frag.glsl")
    .additional_info("draw_fullscreen");

GPU_SHADER_CREATE_INFO(eevee_film_resolve_depth)
    .do_static_compilation(true)
    .uniform_buf(1, "FilmData", "film")
    .sampler(0, ImageType::FLOAT_2D, "data_tx")
    .sampler(1, ImageType::FLOAT_2D, "weight_tx")
    //.fragment_out(0, Type::FLOAT, "gl_FragDepth")
    .additional_info("eevee_shared")
    .fragment_source("eevee_film_resolve_depth_frag.glsl")
    .additional_info("draw_fullscreen");

/** \} */