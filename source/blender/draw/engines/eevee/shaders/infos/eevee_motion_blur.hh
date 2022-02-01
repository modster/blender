
#include "gpu_shader_create_info.hh"

GPU_SHADER_CREATE_INFO(eevee_motion_blur_gather)
    .uniform_buf(0, "MotionBlurData", "mb")
    .uniform_buf(1, "SamplingData", "sampling")
    .sampler(0, ImageType::FLOAT_2D, "color_tx")
    .sampler(1, ImageType::DEPTH_2D, "depth_tx")
    .sampler(2, ImageType::FLOAT_2D, "velocity_tx")
    .sampler(3, ImageType::FLOAT_2D, "tiles_tx")
    .fragment_out(0, Type::VEC4, "out_color")
    .typedef_source("eevee_shader_shared.hh")
    .fragment_source("eevee_motion_blur_gather_frag.glsl")
    .additional_info("draw_fullscreen");

GPU_SHADER_CREATE_INFO(eevee_motion_blur_tiles_dilate)
    .uniform_buf(0, "MotionBlurData", "mb")
    .sampler(0, ImageType::FLOAT_2D, "tiles_tx")
    .fragment_out(0, Type::VEC4, "out_max_motion")
    .typedef_source("eevee_shader_shared.hh")
    .fragment_source("eevee_motion_blur_tiles_dilate_frag.glsl")
    .additional_info("draw_fullscreen");

GPU_SHADER_CREATE_INFO(eevee_motion_blur_tiles_flatten)
    .uniform_buf(0, "MotionBlurData", "mb")
    .sampler(0, ImageType::FLOAT_2D, "tiles_tx")
    .fragment_out(0, Type::VEC4, "out_max_motion")
    .typedef_source("eevee_shader_shared.hh")
    .fragment_source("eevee_motion_blur_tiles_flatten_frag.glsl")
    .additional_info("draw_fullscreen");
