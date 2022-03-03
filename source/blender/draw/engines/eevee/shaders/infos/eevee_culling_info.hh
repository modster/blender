
/* FIXME(@fclem): This file is included inside the gpu module. We have to workaround to include
 * eevee header. */
#include "../../draw/engines/eevee/eevee_defines.hh"

#include "gpu_shader_create_info.hh"

/* -------------------------------------------------------------------- */
/** \name Culling
 * \{ */

GPU_SHADER_CREATE_INFO(eevee_culling_select)
    .do_static_compilation(true)
    .additional_info("eevee_shared", "draw_view")
    .local_group_size(CULLING_BATCH_SIZE)
    .storage_buf(0, Qualifier::READ, "LightData", "lights_buf[]")
    .storage_buf(1, Qualifier::READ_WRITE, "CullingData", "lights_cull_buf")
    .storage_buf(2, Qualifier::READ_WRITE, "uint", "keys_buf[]")
    .compute_source("eevee_culling_select_comp.glsl");

GPU_SHADER_CREATE_INFO(eevee_culling_sort)
    .do_static_compilation(true)
    .additional_info("eevee_shared", "draw_view")
    .local_group_size(CULLING_BATCH_SIZE)
    .storage_buf(0, Qualifier::READ, "LightData", "lights_buf[]")
    .storage_buf(1, Qualifier::READ_WRITE, "CullingData", "lights_cull_buf")
    .storage_buf(2, Qualifier::READ, "uint", "keys_buf[]")
    .storage_buf(3, Qualifier::WRITE, "CullingZBin", "lights_zbin_buf[]")
    .storage_buf(4, Qualifier::WRITE, "LightData", "out_lights_buf[]")
    .fragment_out(0, Type::VEC4, "out_velocity_camera")
    .fragment_out(1, Type::VEC4, "out_velocity_view")
    .compute_source("eevee_culling_sort_comp.glsl");

GPU_SHADER_CREATE_INFO(eevee_culling_tile)
    .do_static_compilation(true)
    .additional_info("eevee_shared", "draw_view")
    .local_group_size(1024)
    .storage_buf(0, Qualifier::READ, "LightData", "lights_buf[]")
    .storage_buf(1, Qualifier::READ, "CullingData", "lights_cull_buf")
    .storage_buf(2, Qualifier::WRITE, "CullingWord", "lights_tile_buf[]")
    .compute_source("eevee_culling_tile_comp.glsl");

GPU_SHADER_CREATE_INFO(eevee_light_data)
    .storage_buf(0, Qualifier::READ, "LightData", "lights_buf[]")
    .storage_buf(1, Qualifier::READ, "CullingZBin", "lights_zbin_buf[]")
    .storage_buf(2, Qualifier::READ, "CullingData", "lights_cull_buf")
    .storage_buf(3, Qualifier::READ, "CullingWord", "lights_tile_buf[]");

/** \} */

/* -------------------------------------------------------------------- */
/** \name Debug
 * \{ */

GPU_SHADER_CREATE_INFO(eevee_culling_debug)
    .do_static_compilation(true)
    .sampler(0, ImageType::FLOAT_2D, "hiz_tx")
    .fragment_out(0, Type::VEC4, "out_debug_color")
    .additional_info("eevee_shared", "draw_view")
    .fragment_source("eevee_culling_debug_frag.glsl")
    .additional_info("draw_fullscreen", "eevee_light_data");

/** \} */
