
/* FIXME(@fclem): This file is included inside the gpu module. We have to workaround to include
 * eevee header. */
#include "../../draw/engines/eevee/eevee_defines.hh"

#include "gpu_shader_create_info.hh"

/* -------------------------------------------------------------------- */
/** \name Ray Generation
 * \{ */

GPU_SHADER_CREATE_INFO(eevee_raytrace_raygen)
    .do_static_compilation(true)
    .local_group_size(RAYTRACE_GROUP_SIZE, RAYTRACE_GROUP_SIZE)
    .additional_info("eevee_shared")
    .storage_buf(0, Qualifier::READ_WRITE, "DispatchIndirectCommand", "dispatch_diffuse_buf")
    .storage_buf(1, Qualifier::READ_WRITE, "DispatchIndirectCommand", "dispatch_reflect_buf")
    .storage_buf(2, Qualifier::READ_WRITE, "DispatchIndirectCommand", "dispatch_refract_buf")
    .storage_buf(4, Qualifier::WRITE, "uint", "tiles_diffuse_buf[]")
    .storage_buf(5, Qualifier::WRITE, "uint", "tiles_reflect_buf[]")
    .storage_buf(6, Qualifier::WRITE, "uint", "tiles_refract_buf[]")
    .uniform_buf(3, "RaytraceBufferData", "raytrace_buffer_buf")
    .sampler(0, ImageType::FLOAT_2D, "gbuf_transmit_data_tx")
    .sampler(1, ImageType::FLOAT_2D, "gbuf_transmit_normal_tx")
    .sampler(2, ImageType::FLOAT_2D, "gbuf_reflection_normal_tx")
    .sampler(3, ImageType::FLOAT_2D, "depth_tx")
    .sampler(4, ImageType::UINT_2D, "stencil_tx")
    .image(0, GPU_RGBA16F, Qualifier::WRITE, ImageType::FLOAT_2D, "out_ray_data_diffuse")
    .image(1, GPU_RGBA16F, Qualifier::WRITE, ImageType::FLOAT_2D, "out_ray_data_reflect")
    .image(2, GPU_RGBA16F, Qualifier::WRITE, ImageType::FLOAT_2D, "out_ray_data_refract")
    .compute_source("eevee_raytrace_raygen_comp.glsl")
    .additional_info("draw_view", "eevee_utility_texture", "eevee_sampling_data");

/** \} */

/* -------------------------------------------------------------------- */
/** \name Screen Space Raytracing
 * \{ */

GPU_SHADER_CREATE_INFO(eevee_raytrace_dispatch)
    .do_static_compilation(true)
    .local_group_size(1)
    .additional_info("eevee_shared")
    .storage_buf(0, Qualifier::READ_WRITE, "DispatchIndirectCommand", "dispatch_buf")
    .compute_source("eevee_raytrace_dispatch_comp.glsl");

GPU_SHADER_CREATE_INFO(eevee_raytrace_screen)
    .local_group_size(RAYTRACE_GROUP_SIZE, RAYTRACE_GROUP_SIZE)
    .additional_info("eevee_shared", "draw_view", "draw_debug_print")
    .storage_buf(0, Qualifier::READ, "DispatchIndirectCommand", "dispatch_buf")
    .storage_buf(2, Qualifier::READ, "uint", "tiles_buf[]")
    .uniform_buf(4, "HiZData", "hiz_buf")
    .uniform_buf(5, "RaytraceBufferData", "raytrace_buffer_buf")
    .uniform_buf(6, "RaytraceData", "raytrace_buf")
    .sampler(0, ImageType::FLOAT_2D, "radiance_tx")
    .sampler(1, ImageType::FLOAT_2D, "hiz_tx")
    .sampler(2, ImageType::FLOAT_2D, "depth_tx")
    .image(0, GPU_RGBA16F, Qualifier::READ_WRITE, ImageType::FLOAT_2D, "inout_ray_data")
    .image(1, GPU_RGBA16F, Qualifier::WRITE, ImageType::FLOAT_2D, "out_ray_radiance")
    .compute_source("eevee_raytrace_screen_comp.glsl")
    .additional_info("eevee_lightprobe_data", "eevee_sampling_data");

GPU_SHADER_CREATE_INFO(eevee_raytrace_screen_reflect)
    .do_static_compilation(true)
    .define("DO_REFLECTION", "true")
    .define("DO_REFRACTION", "false")
    .additional_info("eevee_raytrace_screen");

GPU_SHADER_CREATE_INFO(eevee_raytrace_screen_refract)
    .do_static_compilation(true)
    .define("DO_REFLECTION", "false")
    .define("DO_REFRACTION", "true")
    .additional_info("eevee_raytrace_screen");

/** \} */

/* -------------------------------------------------------------------- */
/** \name De-noising
 * \{ */

GPU_SHADER_CREATE_INFO(eevee_raytrace_denoise)
    .local_group_size(RAYTRACE_GROUP_SIZE, RAYTRACE_GROUP_SIZE)
    .additional_info("eevee_shared", "draw_view")
    .storage_buf(0, Qualifier::READ, "DispatchIndirectCommand", "dispatch_buf")
    .storage_buf(2, Qualifier::READ, "uint", "tiles_buf[]")
    .uniform_buf(1, "RaytraceBufferData", "raytrace_buffer_buf")
    .uniform_buf(2, "RaytraceData", "raytrace_buf")
    .sampler(0, ImageType::FLOAT_2D, "gbuf_data_tx")
    .sampler(1, ImageType::FLOAT_2D, "gbuf_normal_tx")
    .sampler(2, ImageType::DEPTH_2D, "depth_tx")
    .sampler(3, ImageType::UINT_2D, "stencil_tx")
    .sampler(4, ImageType::FLOAT_2D, "ray_data_tx")
    .sampler(5, ImageType::FLOAT_2D, "ray_radiance_tx")
    .sampler(6, ImageType::FLOAT_2D, "ray_history_tx")
    .sampler(7, ImageType::FLOAT_2D, "ray_variance_tx")
    .image(0, GPU_R11F_G11F_B10F, Qualifier::READ_WRITE, ImageType::FLOAT_2D, "out_history_img")
    .image(1, GPU_R8, Qualifier::READ_WRITE, ImageType::FLOAT_2D, "out_variance_img")
    .compute_source("eevee_raytrace_denoise_comp.glsl");

GPU_SHADER_CREATE_INFO(eevee_raytrace_denoise_diffuse)
    .do_static_compilation(true)
    .define("CLOSURE_FLAG", "CLOSURE_DIFFUSE")
    .define("DENOISE_DIFFUSE")
    .additional_info("eevee_raytrace_denoise");

GPU_SHADER_CREATE_INFO(eevee_raytrace_denoise_reflect)
    .do_static_compilation(true)
    .define("CLOSURE_FLAG", "CLOSURE_REFLECTION")
    .define("DENOISE_REFLECTION")
    .additional_info("eevee_raytrace_denoise");

GPU_SHADER_CREATE_INFO(eevee_raytrace_denoise_refract)
    .do_static_compilation(true)
    .define("CLOSURE_FLAG", "CLOSURE_REFRACTION")
    .define("DENOISE_REFRACTION")
    .additional_info("eevee_raytrace_denoise");

/** \} */

/* -------------------------------------------------------------------- */
/** \name Data
 * \{ */

GPU_SHADER_CREATE_INFO(eevee_raytrace_data)
    .define("SCREEN_RAYTRACE")
    .uniform_buf(0, "RaytraceData", "raytrace_info_buf")
    .uniform_buf(3, "HiZData", "hiz_buf")
    .sampler(0, ImageType::FLOAT_2D, "hiz_tx")
    .sampler(1, ImageType::FLOAT_2D, "radiance_tx");

/** \} */
