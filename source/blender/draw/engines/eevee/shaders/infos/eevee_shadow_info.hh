
#include "gpu_shader_create_info.hh"

/* -------------------------------------------------------------------- */
/** \name Shadow pipeline
 * \{ */

GPU_SHADER_CREATE_INFO(eevee_shadow_page_alloc)
    .do_static_compilation(true)
    .typedef_source("eevee_shader_shared.hh")
    .local_group_size(SHADOW_TILEMAP_RES, SHADOW_TILEMAP_RES)
    .storage_buf(1, Qualifier::RESTRICT__READ_ONLY, "ShadowTileMapData", "tilemaps[]")
    .storage_buf(1, Qualifier::RESTRICT, "uint", "free_page_owners[]")
    .storage_buf(3, Qualifier::RESTRICT, "ShadowPagesInfoData", "infos")
    .image(0, GPU_R32UI, Qualifier::RESTRICT, ImageType::UINT_2D, "tilemaps_img")
    .image(1, GPU_R32I, Qualifier::RESTRICT__WRITE_ONLY, ImageType::INT_2D, "tilemap_rects_img")
    .compute_source("eevee_shadow_page_alloc_comp.glsl");

GPU_SHADER_CREATE_INFO(eevee_shadow_page_copy)
    .do_static_compilation(true)
    .typedef_source("eevee_shader_shared.hh")
    .local_group_size(SHADOW_PAGE_COPY_GROUP_SIZE, SHADOW_PAGE_COPY_GROUP_SIZE)
    .sampler(0, ImageType::DEPTH_2D, "render_tx")
    .sampler(1, ImageType::UINT_2D, "tilemaps_tx")
    /* TODO(fclem): 16bit format. */
    .image(0, GPU_R32F, Qualifier::RESTRICT__WRITE_ONLY, ImageType::FLOAT_2D, "out_atlas_img")
    .push_constant(Type::INT, "tilemap_index")
    .push_constant(Type::INT, "tilemap_lod")
    .compute_source("eevee_shadow_page_copy_comp.glsl");

GPU_SHADER_CREATE_INFO(eevee_shadow_page_defrag)
    .do_static_compilation(true)
    .typedef_source("eevee_shader_shared.hh")
    .local_group_size(1)
    .storage_buf(0, Qualifier::RESTRICT, "uint", "free_page_owners[]")
    .storage_buf(1, Qualifier::RESTRICT, "ShadowPagesInfoData", "infos")
    .image(0, GPU_R32UI, Qualifier::RESTRICT, ImageType::UINT_2D, "tilemaps_img")
    .compute_source("eevee_shadow_page_defrag_comp.glsl");

GPU_SHADER_CREATE_INFO(eevee_shadow_page_free)
    .do_static_compilation(true)
    .typedef_source("eevee_shader_shared.hh")
    .local_group_size(SHADOW_TILEMAP_RES, SHADOW_TILEMAP_RES)
    .storage_buf(0, Qualifier::RESTRICT, "uint", "free_page_owners[]")
    .storage_buf(1, Qualifier::RESTRICT, "ShadowPagesInfoData", "infos")
    .storage_buf(2, Qualifier::RESTRICT__READ_ONLY, "ShadowTileMapData", "tilemaps[]")
    .image(0, GPU_R32UI, Qualifier::RESTRICT, ImageType::UINT_2D, "tilemaps_img")
    .compute_source("eevee_shadow_page_free_comp.glsl");

GPU_SHADER_CREATE_INFO(eevee_shadow_page_init)
    .do_static_compilation(true)
    .typedef_source("eevee_shader_shared.hh")
    .local_group_size(SHADOW_PAGE_PER_ROW)
    .storage_buf(0, Qualifier::RESTRICT__WRITE_ONLY, "uint", "free_page_owners[]")
    .storage_buf(1, Qualifier::RESTRICT__WRITE_ONLY, "ShadowPagesInfoData", "infos")
    .image(0, GPU_R32UI, Qualifier::RESTRICT, ImageType::UINT_2D, "tilemaps_img")
    .compute_source("eevee_shadow_page_init_comp.glsl");

GPU_SHADER_CREATE_INFO(eevee_shadow_page_mark)
    .do_static_compilation(true)
    .sampler(0, ImageType::UINT_2D, "tilemaps_tx")
    .push_constant(Type::INT, "tilemap_index")
    .push_constant(Type::INT, "tilemap_lod")
    .fragment_source("eevee_depth_clear_frag.glsl")
    .vertex_source("eevee_shadow_page_mark_vert.glsl");

GPU_SHADER_CREATE_INFO(eevee_shadow_tilemap_depth_scan)
    .do_static_compilation(true)
    .local_group_size(SHADOW_DEPTH_SCAN_GROUP_SIZE, SHADOW_DEPTH_SCAN_GROUP_SIZE)
    .image(0, GPU_R32UI, Qualifier::RESTRICT, ImageType::UINT_2D, "tilemaps_img")
    .sampler(0, ImageType::DEPTH_2D, "depth_tx")
    .push_constant(Type::FLOAT, "tilemap_pixel_radius")
    .push_constant(Type::FLOAT, "screen_pixel_radius_inv")
    .compute_source("eevee_shadow_tilemap_depth_scan_comp.glsl")
    .additional_info("eevee_light_data");

GPU_SHADER_CREATE_INFO(eevee_shadow_tilemap_lod_mask)
    .do_static_compilation(true)
    .typedef_source("eevee_shader_shared.hh")
    .local_group_size(SHADOW_TILEMAP_RES, SHADOW_TILEMAP_RES)
    .storage_buf(0, Qualifier::RESTRICT__READ_ONLY, "ShadowTileMapData", "tilemaps[]")
    .image(0, GPU_R32UI, Qualifier::RESTRICT, ImageType::UINT_2D, "tilemaps_img")
    .compute_source("eevee_shadow_tilemap_lod_mask_comp.glsl");

GPU_SHADER_CREATE_INFO(eevee_shadow_tilemap_setup)
    .do_static_compilation(true)
    .typedef_source("eevee_shader_shared.hh")
    .local_group_size(SHADOW_TILEMAP_RES, SHADOW_TILEMAP_RES)
    .storage_buf(0, Qualifier::RESTRICT, "uint", "free_page_owners[]")
    .storage_buf(1, Qualifier::RESTRICT, "ShadowPagesInfoData", "infos")
    .storage_buf(2, Qualifier::RESTRICT__READ_ONLY, "ShadowTileMapData", "tilemaps[]")
    .image(0, GPU_R32UI, Qualifier::RESTRICT, ImageType::UINT_2D, "tilemaps_img")
    .push_constant(Type::BOOL, "do_tilemap_setup")
    .compute_source("eevee_shadow_tilemap_setup_comp.glsl");

GPU_SHADER_CREATE_INFO(eevee_shadow_tilemap_tag)
    .typedef_source("eevee_shader_shared.hh")
    .local_group_size(SHADOW_AABB_TAG_GROUP_SIZE)
    .storage_buf(1, Qualifier::RESTRICT__READ_ONLY, "AABB", "aabbs[]")
    .storage_buf(2, Qualifier::RESTRICT__READ_ONLY, "ShadowTileMapData", "tilemaps[]")
    .push_constant(Type::INT, "aabb_len")
    .push_constant(Type::FLOAT, "tilemap_pixel_radius")
    .push_constant(Type::FLOAT, "screen_pixel_radius_inv")
    .compute_source("eevee_shadow_tilemap_tag_comp.glsl");

GPU_SHADER_CREATE_INFO(eevee_shadow_tilemap_tag_usage)
    .do_static_compilation(true)
    .define("TAG_USAGE")
    .additional_info("eevee_shadow_tilemap_tag");

GPU_SHADER_CREATE_INFO(eevee_shadow_tilemap_tag_update)
    .do_static_compilation(true)
    .define("TAG_UPDATE")
    .additional_info("eevee_shadow_tilemap_tag");

GPU_SHADER_CREATE_INFO(eevee_shadow_tilemap_visibility)
    .do_static_compilation(true)
    .typedef_source("eevee_shader_shared.hh")
    .local_group_size(SHADOW_TILEMAP_RES, SHADOW_TILEMAP_RES)
    .storage_buf(2, Qualifier::RESTRICT__READ_ONLY, "ShadowTileMapData", "tilemaps[]")
    .image(0, GPU_R32UI, Qualifier::RESTRICT, ImageType::UINT_2D, "tilemaps_img")
    .push_constant(Type::FLOAT, "tilemap_pixel_radius")
    .push_constant(Type::FLOAT, "screen_pixel_radius_inv")
    .compute_source("eevee_shadow_tilemap_visibility_comp.glsl");

/** \} */

/* -------------------------------------------------------------------- */
/** \name Shadow resources
 * \{ */

GPU_SHADER_CREATE_INFO(eevee_shadow_data)
    .sampler(9, ImageType::FLOAT_2D, "shadow_atlas_tx")
    .sampler(10, ImageType::UINT_2D, "shadow_tilemaps_tx");

/** \} */

/* -------------------------------------------------------------------- */
/** \name Debug
 * \{ */

GPU_SHADER_CREATE_INFO(eevee_shadow_page_debug)
    .do_static_compilation(true)
    .local_group_size(8, 8)
    .storage_buf(1, Qualifier::RESTRICT__READ_ONLY, "uint", "free_page_owners[]")
    .image(0, GPU_R32UI, Qualifier::RESTRICT, ImageType::UINT_2D, "tilemaps_img")
    .image(1, GPU_R32UI, Qualifier::RESTRICT, ImageType::UINT_2D, "debug_img")
    .compute_source("eevee_shadow_page_debug_comp.glsl");

GPU_SHADER_CREATE_INFO(eevee_shadow_debug)
    .do_static_compilation(true)
    .typedef_source("eevee_shader_shared.hh")
    .uniform_buf(0, "ShadowDebugData", "debug")
    .storage_buf(0, Qualifier::RESTRICT__READ_ONLY, "ShadowTileMapData", "tilemaps[]")
    .sampler(0, Type::DEPTH_2D, "depth_tx")
    .sampler(1, Type::FLOAT_2D, "atlas_tx")
    .sampler(2, Type::UINT_2D, "tilemaps_tx")
    .sampler(3, Type::UINT_2D, "debug_page_tx")
    .fragment_out(0, Type::VEC4, "out_color_add", DualBlend::SRC_0)
    .fragment_out(0, Type::VEC4, "out_color_mul", DualBlend::SRC_1)
    .fragment_source("eevee_shadow_debug_frag.glsl")
    .additional_info("draw_fullscreen");

/** \} */
