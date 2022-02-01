
#include "gpu_shader_create_info.hh"

/* -------------------------------------------------------------------- */
/** \name Culling
 * \{ */

GPU_SHADER_CREATE_INFO(eevee_culling_select)
    .local_group_size(CULLING_BATCH_SIZE)
    .storage_buf(0, Qualifier::RESTRICT__READ_ONLY, "LightData", "lights[]")
    .storage_buf(1, Qualifier::RESTRICT, "CullingData", "culling")
    .storage_buf(2, Qualifier::RESTRICT, "uint", "keys[]")
    .typedef_source("eevee_shader_shared.hh")
    .compute_source("eevee_culling_select_comp.glsl");

GPU_SHADER_CREATE_INFO(eevee_culling_sort)
    .local_group_size(CULLING_BATCH_SIZE)
    .storage_buf(0, Qualifier::RESTRICT__READ_ONLY, "LightData", "lights[]")
    .storage_buf(1, Qualifier::RESTRICT, "CullingData", "culling")
    .storage_buf(2, Qualifier::RESTRICT__READ_ONLY, "uint", "keys[]")
    .storage_buf(3, Qualifier::RESTRICT__WRITE_ONLY, "CullingZBin", "out_zbins[]")
    .storage_buf(4, Qualifier::RESTRICT__WRITE_ONLY, "LightData", "out_lights[]")
    .sampler(0, ImageType::DEPTH_2D, "depth_tx")
    .fragment_out(0, Type::VEC4, "out_velocity_camera")
    .fragment_out(1, Type::VEC4, "out_velocity_view")
    .typedef_source("eevee_shader_shared.hh")
    .compute_source("eevee_culling_sort_comp.glsl");

GPU_SHADER_CREATE_INFO(eevee_culling_tile)
    .local_group_size(1024)
    .storage_buf(0, Qualifier::RESTRICT__READ_ONLY, "LightData", "lights[]")
    .storage_buf(1, Qualifier::RESTRICT__READ_ONLY, "CullingData", "culling")
    .storage_buf(2, Qualifier::RESTRICT__WRITE_ONLY, "CullingWord", "culling_words[]")
    .typedef_source("eevee_shader_shared.hh")
    .compute_source("eevee_culling_tile_comp.glsl");

GPU_SHADER_CREATE_INFO(eevee_light_data)
    .typedef_source("eevee_shader_shared.hh")
    .storage_buf(0, Qualifier::RESTRICT__READ_ONLY, "LightData", "lights[]")
    .storage_buf(1, Qualifier::RESTRICT__READ_ONLY, "CullingZBin", "lights_zbins[]")
    .storage_buf(2, Qualifier::RESTRICT__READ_ONLY, "CullingData", "light_culling")
    .storage_buf(3, Qualifier::RESTRICT__READ_ONLY, "CullingWord", "lights_culling_words[]");

/** \} */

/* -------------------------------------------------------------------- */
/** \name Debug
 * \{ */

GPU_SHADER_CREATE_INFO(eevee_culling_debug)
    .sampler(0, ImageType::DEPTH_2D, "depth_tx")
    .fragment_out(0, Type::VEC4, "out_debug_color")
    .typedef_source("eevee_shader_shared.hh")
    .fragment_source("eevee_culling_debug_frag.glsl")
    .additional_info("draw_fullscreen", "eevee_light_data");

/** \} */
