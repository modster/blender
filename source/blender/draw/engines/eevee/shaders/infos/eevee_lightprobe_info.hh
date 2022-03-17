
#include "gpu_shader_create_info.hh"

/* -------------------------------------------------------------------- */
/** \name Lightprobe Display
 * \{ */

GPU_SHADER_INTERFACE_INFO(eevee_lightprobe_display_iface, "interp")
    .smooth(Type::VEC3, "P")
    .smooth(Type::VEC2, "coord")
    .flat(Type::INT, "samp");

GPU_SHADER_CREATE_INFO(eevee_lightprobe_display_cubemap)
    .do_static_compilation(true)
    .additional_info("eevee_shared", "draw_view")
    .uniform_buf(1, "LightProbeInfoData", "probes_buf")
    .uniform_buf(2, "CubemapData", "cubes_buf[CULLING_ITEM_BATCH]")
    .sampler(0, ImageType::FLOAT_CUBE_ARRAY, "lightprobe_cube_tx")
    .vertex_out(eevee_lightprobe_display_iface)
    .fragment_out(0, Type::VEC4, "out_color")
    .vertex_source("eevee_lightprobe_display_cubemap_vert.glsl")
    .fragment_source("eevee_lightprobe_display_cubemap_frag.glsl");

GPU_SHADER_CREATE_INFO(eevee_lightprobe_display_grid)
    .do_static_compilation(true)
    .additional_info("eevee_shared", "draw_view")
    .uniform_buf(1, "GridData", "grids_buf[GRID_MAX]")
    .uniform_buf(2, "LightProbeInfoData", "probes_buf")
    .sampler(0, ImageType::FLOAT_2D_ARRAY, "lightprobe_grid_tx")
    .push_constant(Type::INT, "grid_id")
    .vertex_out(eevee_lightprobe_display_iface)
    .fragment_out(0, Type::VEC4, "out_color")
    .vertex_source("eevee_lightprobe_display_grid_vert.glsl")
    .fragment_source("eevee_lightprobe_display_grid_frag.glsl");

/** \} */

/* -------------------------------------------------------------------- */
/** \name Lightprobe filter
 * \{ */
GPU_SHADER_CREATE_INFO(eevee_lightprobe_filter_cubemap)
    .do_static_compilation(true)
    .local_group_size(16, 16)
    .uniform_buf(1, "LightProbeFilterData", "filter_buf")
    .sampler(0, ImageType::FLOAT_2D_ARRAY, "radiance_tx")
    .image(0, GPU_RGBA16F, Qualifier::WRITE, ImageType::FLOAT_2D_ARRAY, "out_lvl0")
    .image(1, GPU_RGBA16F, Qualifier::WRITE, ImageType::FLOAT_2D_ARRAY, "out_lvl1")
    .image(2, GPU_RGBA16F, Qualifier::WRITE, ImageType::FLOAT_2D_ARRAY, "out_lvl2")
    .image(3, GPU_RGBA16F, Qualifier::WRITE, ImageType::FLOAT_2D_ARRAY, "out_lvl3")
    .image(4, GPU_RGBA16F, Qualifier::WRITE, ImageType::FLOAT_2D_ARRAY, "out_lvl4")
    .image(5, GPU_RGBA16F, Qualifier::WRITE, ImageType::FLOAT_2D_ARRAY, "out_lvl5")
    .additional_info("eevee_shared")
    .compute_source("eevee_lightprobe_filter_cubemap_comp.glsl");

GPU_SHADER_CREATE_INFO(eevee_lightprobe_filter_diffuse)
    .do_static_compilation(true)
    .local_group_size(3, 2)
    .uniform_buf(1, "LightProbeFilterData", "filter_buf")
    .sampler(0, ImageType::FLOAT_CUBE, "radiance_tx")
    .image(0, GPU_RGBA8, Qualifier::WRITE, ImageType::FLOAT_2D_ARRAY, "out_irradiance_cache_img")
    .additional_info("eevee_shared")
    .compute_source("eevee_lightprobe_filter_diffuse_comp.glsl");

GPU_SHADER_CREATE_INFO(eevee_lightprobe_filter_visibility)
    .do_static_compilation(true)
    .local_group_size(LIGHTPROBE_FILTER_VIS_GROUP_SIZE, LIGHTPROBE_FILTER_VIS_GROUP_SIZE)
    .uniform_buf(1, "LightProbeFilterData", "filter_buf")
    .sampler(0, ImageType::DEPTH_CUBE, "depth_tx")
    .image(0, GPU_RGBA8, Qualifier::WRITE, ImageType::FLOAT_2D_ARRAY, "out_visibility_img")
    .additional_info("eevee_shared", "draw_view")
    .compute_source("eevee_lightprobe_filter_visibility_comp.glsl");

/** \} */

/* -------------------------------------------------------------------- */
/** \name Lightprobe data
 * \{ */

GPU_SHADER_CREATE_INFO(eevee_lightprobe_data)
    .uniform_buf(1, "GridData", "grids_buf[GRID_MAX]")
    .uniform_buf(2, "CubemapData", "cubes_buf[CULLING_ITEM_BATCH]")
    .uniform_buf(3, "LightProbeInfoData", "probes_buf")
    .sampler(11, ImageType::FLOAT_2D_ARRAY, "lightprobe_grid_tx")
    .sampler(12, ImageType::FLOAT_CUBE_ARRAY, "lightprobe_cube_tx");

/** \} */
