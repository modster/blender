
#include "gpu_shader_create_info.hh"

/* -------------------------------------------------------------------- */
/** \name Direct Lighting
 * \{ */

GPU_SHADER_CREATE_INFO(eevee_utility_texture).sampler(8, ImageType::FLOAT_2D_ARRAY, "utility_tx");

GPU_SHADER_CREATE_INFO(eevee_deferred_direct)
    .do_static_compilation(true)
    .auto_resource_location(true)
    .sampler(0, ImageType::FLOAT_2D, "hiz_tx")
    .sampler(1, ImageType::FLOAT_2D, "emission_data_tx")
    .sampler(2, ImageType::FLOAT_2D, "transmit_color_tx")
    .sampler(3, ImageType::FLOAT_2D, "transmit_normal_tx")
    .sampler(4, ImageType::FLOAT_2D, "transmit_data_tx")
    .sampler(5, ImageType::FLOAT_2D, "reflect_color_tx")
    .sampler(6, ImageType::FLOAT_2D, "reflect_normal_tx")
    .fragment_out(0, Type::VEC4, "out_combined")
    .fragment_out(1, Type::VEC4, "out_diffuse")
    .fragment_out(2, Type::VEC3, "out_specular")
    .additional_info("eevee_shared", "draw_view")
    .fragment_source("eevee_deferred_direct_frag.glsl")
    .additional_info("draw_fullscreen",
                     "eevee_transmittance_data",
                     "eevee_utility_texture",
                     "eevee_lightprobe_data",
                     "eevee_light_data",
                     "eevee_shadow_data",
                     "eevee_sampling_data");

/** \} */

/* -------------------------------------------------------------------- */
/** \name Holdout
 * \{ */

GPU_SHADER_CREATE_INFO(eevee_deferred_holdout)
    .do_static_compilation(true)
    .sampler(0, ImageType::FLOAT_2D, "combined_tx")
    .sampler(1, ImageType::FLOAT_2D, "transparency_data_tx")
    .fragment_out(5, Type::VEC3, "out_holdout")
    .additional_info("eevee_shared")
    .fragment_source("eevee_deferred_holdout_frag.glsl")
    .additional_info("draw_fullscreen", "draw_view");

/** \} */

/* -------------------------------------------------------------------- */
/** \name Transparency
 * \{ */

GPU_SHADER_CREATE_INFO(eevee_deferred_transparent)
    .do_static_compilation(true)
    .sampler(0, ImageType::UINT_2D, "volume_data_tx")
    .sampler(1, ImageType::FLOAT_2D, "transparency_data_tx")
    .fragment_out(0, Type::VEC4, "out_combined")
    .fragment_out(1, Type::VEC3, "out_diffuse")
    .fragment_out(2, Type::VEC3, "out_specular")
    .fragment_out(3, Type::VEC3, "out_volume")
    .fragment_out(4, Type::VEC3, "out_background")
    .fragment_out(5, Type::VEC3, "out_holdout")
    .additional_info("eevee_shared")
    .fragment_source("eevee_deferred_transparent_frag.glsl")
    .additional_info("draw_fullscreen", "draw_view", "eevee_light_data");

/** \} */

/* -------------------------------------------------------------------- */
/** \name Volume
 * \{ */

GPU_SHADER_CREATE_INFO(eevee_deferred_volume)
    .do_static_compilation(true)
    .sampler(0, ImageType::FLOAT_2D, "transparency_data_tx")
    .sampler(1, ImageType::UINT_2D, "volume_data_tx")
    .fragment_out(0, Type::VEC4, "out_combined")
    .fragment_out(1, Type::VEC3, "out_volume")
    .additional_info("eevee_shared")
    .fragment_source("eevee_deferred_volume_frag.glsl")
    .additional_info("draw_fullscreen",
                     "draw_view",
                     "eevee_utility_texture",
                     "eevee_light_data",
                     "eevee_shadow_data");

/** \} */
