
#include "gpu_shader_create_info.hh"

GPU_SHADER_CREATE_INFO(eevee_transmittance_data)
    .sampler(0, ImageType::FLOAT_1D, "sss_transmittance_tx");

GPU_SHADER_CREATE_INFO(eevee_subsurface_eval)
    .uniform_buf(0, "SubsurfaceData", "sss")
    .uniform_buf(1, "HiZData", "hiz")
    .sampler(0, ImageType::FLOAT_2D, "hiz_tx")
    .sampler(1, ImageType::FLOAT_2D, "radiance_tx")
    .sampler(2, ImageType::FLOAT_2D, "transmit_color_tx")
    .sampler(3, ImageType::FLOAT_2D, "transmit_normal_tx")
    .sampler(4, ImageType::FLOAT_2D, "transmit_data_tx")
    .fragment_out(0, Type::VEC4, "out_combined")
    /* TODO(fclem) Output to diffuse pass without feedback loop. */
    .additional_info("draw_fullscreen");
