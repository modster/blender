
#include "gpu_shader_create_info.hh"

GPU_SHADER_CREATE_INFO(eevee_motion_blur_gather)
    .do_static_compilation(true)
    .additional_info("eevee_shared", "draw_view")
    .uniform_buf(1, "SamplingData", "sampling_buf")
    .uniform_buf(2, "MotionBlurData", "mb_buf")
    .sampler(0, ImageType::FLOAT_2D, "color_tx")
    .sampler(1, ImageType::DEPTH_2D, "depth_tx")
    .sampler(2, ImageType::FLOAT_2D, "velocity_tx")
    .sampler(3, ImageType::FLOAT_2D, "tiles_tx")
    .fragment_out(0, Type::VEC4, "out_color")
    .fragment_source("eevee_motion_blur_gather_frag.glsl")
    .additional_info("draw_fullscreen");

GPU_SHADER_CREATE_INFO(eevee_motion_blur_tiles_dilate)
    .do_static_compilation(true)
    .uniform_buf(1, "MotionBlurData", "mb_buf")
    .sampler(0, ImageType::FLOAT_2D, "tiles_tx")
    .fragment_out(0, Type::VEC4, "out_max_motion")
    .additional_info("eevee_shared")
    .fragment_source("eevee_motion_blur_tiles_dilate_frag.glsl")
    .additional_info("draw_fullscreen");

GPU_SHADER_CREATE_INFO(eevee_motion_blur_tiles_flatten)
    .do_static_compilation(true)
    .uniform_buf(1, "MotionBlurData", "mb_buf")
    .sampler(0, ImageType::FLOAT_2D, "tiles_tx")
    .sampler(2, ImageType::FLOAT_2D, "velocity_tx")
    .fragment_out(0, Type::VEC4, "out_max_motion")
    .additional_info("eevee_shared")
    .fragment_source("eevee_motion_blur_tiles_flatten_frag.glsl")
    .additional_info("draw_fullscreen");
