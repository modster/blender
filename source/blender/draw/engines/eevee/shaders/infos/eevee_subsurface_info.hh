
#include "gpu_shader_create_info.hh"

GPU_SHADER_CREATE_INFO(eevee_transmittance_data)
    .define("SSS_TRANSMITTANCE")
    .sampler(0, ImageType::FLOAT_1D, "sss_transmittance_tx");

GPU_SHADER_CREATE_INFO(eevee_subsurface_eval)
    .do_static_compilation(true)
    .additional_info("eevee_shared")
    .uniform_buf(2, "SubsurfaceData", "sss_buf")
    .uniform_buf(1, "HiZData", "hiz_buf")
    .sampler(0, ImageType::FLOAT_2D, "hiz_tx")
    .sampler(1, ImageType::FLOAT_2D, "radiance_tx")
    .sampler(2, ImageType::FLOAT_2D, "transmit_color_tx")
    .sampler(3, ImageType::FLOAT_2D, "transmit_normal_tx")
    .sampler(4, ImageType::FLOAT_2D, "transmit_data_tx")
    .fragment_out(0, Type::VEC4, "out_combined")
    .fragment_source("eevee_subsurface_eval_frag.glsl")
    /* TODO(fclem) Output to diffuse pass without feedback loop. */
    .additional_info("draw_fullscreen", "draw_view");
