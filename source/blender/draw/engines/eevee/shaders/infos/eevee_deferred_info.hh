
#include "gpu_shader_create_info.hh"

/* -------------------------------------------------------------------- */
/** \name Direct Lighting
 * \{ */

GPU_SHADER_CREATE_INFO(eevee_utility_texture).sampler(8, ImageType::FLOAT_2D_ARRAY, "utility_tx");

#define image_out(slot, qualifier, format, name) \
  image(slot, format, qualifier, ImageType::FLOAT_2D, name, Frequency::PASS)

GPU_SHADER_CREATE_INFO(eevee_deferred_eval)
    .do_static_compilation(true)
    .auto_resource_location(true)
    .sampler(0, ImageType::FLOAT_2D, "hiz_tx")
    .sampler(2, ImageType::FLOAT_2D, "transmit_color_tx")
    .sampler(3, ImageType::FLOAT_2D, "transmit_normal_tx")
    .sampler(4, ImageType::FLOAT_2D, "transmit_data_tx")
    .sampler(5, ImageType::FLOAT_2D, "reflect_color_tx")
    .sampler(6, ImageType::FLOAT_2D, "reflect_normal_tx")
    .fragment_out(0, Type::VEC4, "out_combined")
    /* Renderpasses. */
    // .image_out(0, Qualifier::READ_WRITE, GPU_RGBA16F, "rpass_diffuse_light")
    // .image_out(1, Qualifier::READ_WRITE, GPU_RGBA16F, "rpass_specular_light")
    /* Raytracing. */
    // .image_out(2, Qualifier::WRITE, GPU_RGBA16F, "ray_data")
    /* SubSurfaceScattering. */
    // .image_out(3, Qualifier::WRITE, GPU_RGBA16F, "sss_radiance")
    .additional_info("eevee_shared", "draw_view")
    .fragment_source("eevee_deferred_eval_frag.glsl")
    .additional_info("draw_fullscreen",
                     "eevee_transmittance_data",
                     "eevee_utility_texture",
                     //  "eevee_lightprobe_data",
                     "eevee_light_data",
                     "eevee_shadow_data",
                     "eevee_sampling_data");

#undef image_out

/** \} */
