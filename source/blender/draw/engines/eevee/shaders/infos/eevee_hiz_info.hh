
#include "gpu_shader_create_info.hh"

/** NOTE: Read depth format, output color format. */
GPU_SHADER_CREATE_INFO(eevee_hiz_copy)
    .do_static_compilation(true)
    .sampler(0, ImageType::DEPTH_2D, "depth_tx")
    .fragment_out(0, Type::FLOAT, "out_depth")
    .fragment_source("eevee_hiz_copy_frag.glsl")
    .additional_info("draw_fullscreen");

GPU_SHADER_CREATE_INFO(eevee_hiz_downsample)
    .do_static_compilation(true)
    .sampler(0, ImageType::DEPTH_2D, "depth_tx")
    .fragment_out(0, Type::FLOAT, "out_depth")
    .push_constant(Type::VEC2, "texel_size")
    .fragment_source("eevee_hiz_downsample_frag.glsl")
    .additional_info("draw_fullscreen");
