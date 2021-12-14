
#include "gpu_shader_create_info.hh"

GPU_SHADER_CREATE_INFO(workbench_effect_cavity)
    .do_static_compilation(true)
    .fragment_out(0, Type::VEC4, "fragColor")
    .sampler(0, ImageType::FLOAT_2D, "depthBuffer")
    .sampler(1, ImageType::FLOAT_2D, "normalBuffer")
    .sampler(2, ImageType::UINT_2D, "objectIdBuffer")
    .fragment_source("workbench_effect_cavity_frag.glsl")
    .additional_info("draw_fullscreen");
